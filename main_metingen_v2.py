import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# Constanten
LAMBDA     = 530e-9   # m
L          = 0.10     # m
MIN_DIST_S = 0.010
SMOOTH_WIN, SMOOTH_ORD = 5, 3
KLEUREN = ["steelblue", "darkorange", "forestgreen", "mediumpurple", "crimson"]
DRUK_TOLERANTIE_KPA = 0.1  # drukdalingen binnen deze afstand worden gemiddeld

# Elke run bewaart zijn eigen atmosferische druk en meetpunten.
runs = []
huidige_run = None   # {"p_atm": kPa, "metingen": [(p_kPa, ΔN_cumulatief), ...]}
dN_cum = 0


def run_nulpunt():
    """Sla de huidige run op en start een nieuwe run met ΔN = 0."""
    global huidige_run, dN_cum

    if huidige_run is not None and len(huidige_run["metingen"]) > 1:
        runs.append(huidige_run)
        print(f"Run {len(runs)} opgeslagen ({len(huidige_run['metingen'])} meetpunten).")

    p_atm = float(input("Atmosferische druk p_atm (kPa): ").strip())
    huidige_run = {"p_atm": p_atm, "metingen": [(p_atm, 0)]}
    dN_cum = 0
    print(f"Nieuwe run gestart: p_atm = {p_atm} kPa, ΔN = 0.")


def run_oscilloscoop():
    """Lees één oscilloscoop-CSV, tel franjes in een zelfgekozen tijdsbereik."""
    global dN_cum

    if huidige_run is None:
        print("Start eerst een run met optie 0.")
        return

    nummer = input("Newfile nummer: ").strip()
    csv_path = f"Newfile{nummer}.csv"

    meta = pd.read_csv(csv_path, header=None, nrows=2).iloc[1]
    t_start, t_step = float(meta[2]), float(meta[3])
    df = (pd.read_csv(csv_path, skiprows=2, header=None, usecols=[0, 1],
                      names=["seq", "CH1"])
            .apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True))

    if len(df) < 5:
        print("Te weinig geldige samples in de CSV.")
        return

    t = t_start + df["seq"].to_numpy() * t_step
    y = df["CH1"].to_numpy()
    dt = float(np.median(np.diff(t)))

    resultaat = {"n": None, "bevestigd": False}
    staat = {"fase": "selecteer"}
    sel = {}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 7), gridspec_kw={"hspace": 0.5}
    )
    ax_top.plot(t, y, lw=0.5, color="steelblue")
    ax_top.set(title=f"Volledig signaal — {csv_path} | ΔN tot nu: {dN_cum}",
               xlabel="Tijd (s)", ylabel="V")
    ax_top.legend(fontsize=8)
    ax_bot.set(title="Selectie", xlabel="Tijd (s)", ylabel="V")
    info = fig.text(0.5, 0.95, "① Sleep een bereik en druk Enter om pieken te tellen.",
                    ha="center", va="top", fontsize=9, color="darkred")

    def on_select(xmin, xmax):
        if staat["fase"] != "selecteer":
            return
        sel.update(xmin=xmin, xmax=xmax)
        n_samples = int(np.sum((t >= xmin) & (t <= xmax)))
        info.set_text(f"Bereik: {xmin:.3e} → {xmax:.3e} s | {n_samples} samples — druk Enter")
        fig.canvas.draw_idle()

    def on_enter(event):
        if event.key != "enter":
            return

        if staat["fase"] == "selecteer":
            if "xmin" not in sel:
                return
            mask = (t >= sel["xmin"]) & (t <= sel["xmax"])
            t_sel, y_sel = t[mask], y[mask]
            if len(y_sel) < SMOOTH_WIN:
                info.set_text("Selecteer een groter bereik met meer samples.")
                fig.canvas.draw_idle()
                return

            win = min(SMOOTH_WIN, len(y_sel) if len(y_sel) % 2 else len(y_sel) - 1)
            if win <= SMOOTH_ORD:
                info.set_text("Selecteer een groter bereik om te kunnen filteren.")
                fig.canvas.draw_idle()
                return
            y_sm = savgol_filter(y_sel, win, SMOOTH_ORD)
            peaks, _ = find_peaks(
                y_sm,
                prominence=(y_sm.max() - y_sm.min()) * 0.10,
                distance=max(1, int(MIN_DIST_S / dt))
            )
            n = len(peaks)

            ax_bot.cla()
            ax_bot.plot(t_sel, y_sel, lw=0.8, color="lightsteelblue", label="raw")
            ax_bot.plot(t_sel, y_sm, lw=1.2, color="steelblue", label="smooth")
            ax_bot.plot(t_sel[peaks], y_sm[peaks], "rv", ms=8, label=f"maxima n={n}")
            ax_bot.set(title=f"[{sel['xmin']:.3e} – {sel['xmax']:.3e} s] | pieken: {n}",
                       xlabel="Tijd (s)", ylabel="V")
            ax_bot.legend(fontsize=8)
            info.set_text(f"Pieken geteld: {n} — druk Enter om te bevestigen, of selecteer opnieuw.")
            info.set_color("darkorange")

            resultaat.update(n=n, xmin=sel["xmin"], xmax=sel["xmax"],
                             samples=int(mask.sum()))
            staat["fase"] = "check"
            fig.canvas.draw_idle()

        elif staat["fase"] == "check":
            resultaat["bevestigd"] = True
            staat["fase"] = "klaar"
            info.set_text("Bevestigd — venster sluit...")
            info.set_color("darkgreen")
            fig.canvas.draw_idle()
            plt.close(fig)

    def on_select_reset(xmin, xmax):
        if staat["fase"] == "check":
            staat["fase"] = "selecteer"
            info.set_color("darkred")
        on_select(xmin, xmax)

    span = widgets.SpanSelector(
        ax_top, on_select_reset, "horizontal", useblit=True,
        interactive=True, drag_from_anywhere=True,
        props=dict(alpha=0.3, facecolor="orange")
    )
    fig.canvas.mpl_connect("key_press_event", on_enter)
    plt.show()

    if not resultaat["bevestigd"]:
        print("Niet bevestigd; meting overgeslagen.")
        return

    n_auto = resultaat["n"]
    print(f"\nBereik: {resultaat['xmin']:.4e} - {resultaat['xmax']:.4e} s | "
          f"samples: {resultaat['samples']} | pieken automatisch: {n_auto}")
    raw = input(f"Aantal franjes [{n_auto}] (Enter = overnemen): ").strip()
    n_stap = int(raw) if raw else n_auto
    p = float(input("Druk p op dit moment (kPa): "))

    vorige_p = huidige_run["metingen"][-1][0]
    if p > vorige_p:
        print("Waarschuwing: de druk is hoger dan bij het vorige meetpunt. "
              "Cumulatief franjes tellen veronderstelt doorgaans dalende druk.")

    dN_cum += n_stap
    huidige_run["metingen"].append((p, dN_cum))
    print(f"Opgeslagen in run {len(runs) + 1}: p={p} kPa, ΔN={dN_cum} | "
          f"meetpunten in run: {len(huidige_run['metingen'])}")


def groepeer_en_middel(raw_points, tolerantie_kpa):
    """Groepeer nabije drukdalingen, met maximaal één punt per run per groep."""
    groepen = []
    for punt in sorted(raw_points, key=lambda item: item[1]):
        run_id, dp_kpa, dN = punt
        kandidaten = []
        for groep in groepen:
            if run_id in groep["runs"]:
                continue
            centrum = np.mean([lid[1] for lid in groep["punten"]])
            afstand = abs(dp_kpa - centrum)
            if afstand <= tolerantie_kpa:
                kandidaten.append((afstand, groep))
        if kandidaten:
            _, beste_groep = min(kandidaten, key=lambda item: item[0])
            beste_groep["punten"].append((run_id, dp_kpa, dN))
            beste_groep["runs"].add(run_id)
        else:
            groepen.append({"punten": [(run_id, dp_kpa, dN)], "runs": {run_id}})

    x_gem = np.array([np.mean([p[1] for p in g["punten"]]) for g in groepen])
    y_gem = np.array([np.mean([p[2] for p in g["punten"]]) for g in groepen])
    n_per_groep = np.array([len(g["punten"]) for g in groepen])
    return x_gem, y_gem, n_per_groep


def run_fit(runs_override=None):
    """Middel overeenkomende drukdalingen en pas één fit door de oorsprong toe."""
    if runs_override is None:
        alle_runs = list(runs)
        if huidige_run is not None and len(huidige_run["metingen"]) > 1:
            alle_runs.append(huidige_run)
    else:
        alle_runs = list(runs_override)

    if not alle_runs:
        print("Nog geen runs met metingen. Start een run en voeg meetpunten toe.")
        return

    raw_points = []
    run_arrays = []
    for i, run in enumerate(alle_runs):
        p_run = np.array([m[0] for m in run["metingen"]], dtype=float)
        dN_run = np.array([m[1] for m in run["metingen"]], dtype=float)
        dp_run_kpa = run["p_atm"] - p_run
        run_arrays.append((p_run, dN_run, dp_run_kpa))
        raw_points.extend((i, float(dp), float(dn))
                          for dp, dn in zip(dp_run_kpa, dN_run))

    dp_mean_kpa, dN_mean, n_per_group = groepeer_en_middel(
        raw_points, DRUK_TOLERANTIE_KPA
    )
    dp_mean_pa = dp_mean_kpa * 1e3

    if len(dp_mean_pa) < 2 or not np.any(np.abs(dp_mean_pa) > 0):
        print("Er zijn minimaal twee drukpunten en minstens één drukverandering nodig.")
        return

    def model_lin(dp_pa, a):
        return a * dp_pa

    popt, pcov = curve_fit(model_lin, dp_mean_pa, dN_mean)
    a = float(popt[0])
    sigma_a = float(np.sqrt(max(pcov[0, 0], 0)))

    dN_pred = model_lin(dp_mean_pa, a)
    ss_res = np.sum((dN_mean - dN_pred) ** 2)
    ss_tot = np.sum((dN_mean - dN_mean.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    p_atm_ref = float(np.mean([run["p_atm"] for run in alle_runs]))
    n_atm_ref = 1 + (a * LAMBDA * p_atm_ref * 1e3) / (2 * L)
    sigma_n_atm = sigma_a * LAMBDA * p_atm_ref * 1e3 / (2 * L)

    print(f"\n{'─' * 48}")
    print(f"Runs meegenomen          : {len(alle_runs)}")
    print(f"Ruwe meetpunten          : {len(raw_points)}")
    print(f"Gemiddelde fitpunten     : {len(dp_mean_pa)}")
    print(f"Middeltolerantie         : ±{DRUK_TOLERANTIE_KPA:.1f} kPa")
    print(f"Helling a                : {a:.4e} ± {sigma_a:.2e} franjes/Pa (1σ)")
    print(f"R² op de gemiddelde punten: {r2:.4f}")
    print(f"n(p_atm gemiddeld)       : {n_atm_ref:.6f} ± {sigma_n_atm:.6f} (1σ)")
    print(f"Referentie lucht         : 1.000293")
    print(f"{'─' * 48}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Ruwe meetpunten per run, ter vergelijking met de gemiddelden.
    for i, (p_run, dN_run, dp_run_kpa) in enumerate(run_arrays):
        kleur = KLEUREN[i % len(KLEUREN)]
        ax1.scatter(dp_run_kpa, dN_run, color=kleur, alpha=0.35,
                    zorder=2, label=f"ruwe run {i + 1}")
        n_run = 1 + (n_atm_ref - 1) * p_run / p_atm_ref
        ax2.scatter(p_run, n_run, color=kleur, alpha=0.55,
                    zorder=2, label=f"run {i + 1}")

    ax1.errorbar(dp_mean_kpa, dN_mean, xerr=None, fmt="ks", ms=5,
                 capsize=2, zorder=4 ,color = "blue" ,label="gemiddelde per drukpunt")

    dp_fit_kpa = np.linspace(min(0, dp_mean_kpa.min()),
                             max(0, dp_mean_kpa.max()), 400)
    dp_fit_pa = dp_fit_kpa * 1e3
    y_fit = model_lin(dp_fit_pa, a)
    sigma_fit = np.abs(dp_fit_pa) * sigma_a

    # De fit is y = a·x; daarom is σ_fit(x) = |x|·σ_a.
    ax1.fill_between(dp_fit_kpa, y_fit - 3 * sigma_fit, y_fit + 3 * sigma_fit,
                     color="lightskyblue", alpha=0.35, label="3σ-band")
    ax1.fill_between(dp_fit_kpa, y_fit - sigma_fit, y_fit + sigma_fit,
                     color="royalblue", alpha=0.35, label="1σ-band")
    ax1.plot(dp_fit_kpa, y_fit, color="red", lw=1.8,
             label=f"fit avg: y = a·x\na = {a:.3e} ± {sigma_a:.1e} franjes/Pa")
    ax1.set_xlabel("Drukdaling p_atm − p (kPa)")
    ax1.set_ylabel("Cumulatief aantal franjes ΔN")
    ax1.set_title("Franjes versus drukdaling")
    ax1.legend(fontsize=7)

    p_fit = np.linspace(0, p_atm_ref, 300)
    n_fit = 1 + (n_atm_ref - 1) * p_fit / p_atm_ref
    sigma_n_fit = sigma_a * LAMBDA * p_fit * 1e3 / (2 * L)
    ax2.fill_between(p_fit, n_fit - 3 * sigma_n_fit, n_fit + 3 * sigma_n_fit,
                     color="lightskyblue", alpha=0.30, label="3σ-band")
    ax2.fill_between(p_fit, n_fit - sigma_n_fit, n_fit + sigma_n_fit,
                     color="royalblue", alpha=0.30, label="1σ-band")
    ax2.plot(p_fit, n_fit, color="red", lw=1.8,
             label=f"n(p) uit fit; n(p_atm)={n_atm_ref:.6f}")
    ax2.yaxis.get_major_formatter().set_useOffset(False)
    ax2.set_xlabel("Druk p (kPa)")
    ax2.set_ylabel("Berekende brekingsindex n")
    ax2.set_title("Brekingsindex versus druk")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def run_voorbeeld():
    """Draai een losse demo met 3 nep-runs van elk 5 punten; echte data blijven intact."""
    p_atm_demo = 101.3
    drukken = [101.3, 96.3, 91.3, 86.3, 81.3]  # kPa; vijf punten inclusief nulpunt
    aantallen = [
        [0, 2, 4, 6, 8],
        [0, 2, 5, 7, 9],
        [0, 3, 5, 7, 10],
    ]
    demo_runs = [
        {"p_atm": p_atm_demo, "metingen": list(zip(drukken, dN))}
        for dN in aantallen
    ]
    print("Voorbeeldfit: 3 nep-runs x 5 datapunten. Je echte meetdata worden niet aangepast.")
    run_fit(demo_runs)


def main():
    while True:
        n_huidig = len(huidige_run["metingen"]) if huidige_run is not None else 0
        p_huidig = huidige_run["p_atm"] if huidige_run is not None else None
        print(f"\n{'─' * 48}")
        print(f"Opgeslagen runs: {len(runs)} | Huidige run: {n_huidig} meetpunten | "
              f"p_atm huidig: {p_huidig} kPa | ΔN_cum={dN_cum}")
        print("0  Nulpunt instellen / nieuwe run starten")
        print("1  Nieuwe oscilloscoop-CSV — pieken tellen")
        print("2  Gemiddelden, gezamenlijke fit en sigma-banden")
        print("3  Voorbeeldfit: 3 nep-runs met elk 5 datapunten")
        print("q  Stoppen")
        keuze = input("Keuze: ").strip().lower()

        if keuze == "0":
            run_nulpunt()
        elif keuze == "1":
            run_oscilloscoop()
        elif keuze == "2":
            run_fit()
        elif keuze == "3":
            run_voorbeeld()
        elif keuze == "q":
            break
        else:
            print("Ongeldige keuze.")
main()