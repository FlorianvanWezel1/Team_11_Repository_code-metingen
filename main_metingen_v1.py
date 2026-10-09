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
TRIGGER_V  = 0.253
MIN_DIST_S = 0.010
SMOOTH_WIN, SMOOTH_ORD = 5, 3

# Experimentdata
p_atm    = None        # kPa, ingesteld bij nulpunt
metingen = []          # lijst van (p in kPa, ΔN cumulatief)
dN_cum   = 0           # lopende teller


# Nulpunt instellen 
def run_nulpunt():
    global p_atm, dN_cum
    raw = input("Atmosferische druk p_atm (kPa): ").strip()
    p_atm  = float(raw)
    dN_cum = 0
    metingen.clear()
    metingen.append((p_atm, 0))
    print(f"Nulpunt ingesteld: p_atm = {p_atm} kPa, ΔN = 0")


# Piekdetectie van CSV 
def run_oscilloscoop():
    global dN_cum

    if p_atm is None:
        print("Stel eerst het nulpunt in (optie 0).")
        return

    nummer   = input("Newfile nummer: ").strip()
    csv_path = f"Newfile{nummer}.csv"

    meta = pd.read_csv(csv_path, header=None, nrows=2).iloc[1]
    t_start, t_step = float(meta[2]), float(meta[3])

    df = (pd.read_csv(csv_path, skiprows=2, header=None, usecols=[0,1], names=["seq","CH1"])
            .apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True))

    t  = t_start + df["seq"].to_numpy() * t_step
    y  = df["CH1"].to_numpy()
    dt = (t[-1] - t[0]) / (len(t) - 1)

    resultaat = {"n": None, "bevestigd": False}
    staat     = {"fase": "selecteer"}

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"hspace": 0.5})
    ax_top.plot(t, y, lw=0.5, color="steelblue")
    ax_top.axhline(TRIGGER_V, color="red", lw=0.8, ls="--", label=f"Trigger {TRIGGER_V} V")
    ax_top.set(title=f"Volledig signaal — {csv_path}  |  ΔN tot nu: {dN_cum}",
               xlabel="Tijd (s)", ylabel="V")
    ax_top.legend(fontsize=8)
    ax_bot.set(title="Selectie", xlabel="Tijd (s)", ylabel="V")
    info = fig.text(0.5, 0.95, "① Sleep bereik en druk Enter om pieken te tellen.",
                    ha="center", va="top", fontsize=9, color="darkred")

    sel = {}

    def on_select(xmin, xmax):
        if staat["fase"] != "selecteer":
            return
        sel.update(xmin=xmin, xmax=xmax)
        n_samples = np.sum((t >= xmin) & (t <= xmax))
        info.set_text(f"Bereik: {xmin:.3e} → {xmax:.3e} s  |  {n_samples} samples  —  druk Enter om te tellen")
        fig.canvas.draw_idle()

    def on_enter(event):
        if event.key != "enter":
            return

        if staat["fase"] == "selecteer":
            if "xmin" not in sel:
                return
            mask  = (t >= sel["xmin"]) & (t <= sel["xmax"])
            t_sel, y_sel = t[mask], y[mask]
            y_sm  = savgol_filter(y_sel, SMOOTH_WIN, SMOOTH_ORD)
            peaks, _ = find_peaks(y_sm, prominence=(y_sm.max()-y_sm.min())*0.10,
                                  distance=max(1, int(MIN_DIST_S/dt)))
            n = len(peaks)

            ax_bot.cla()
            ax_bot.plot(t_sel, y_sel, lw=0.8, color="lightsteelblue", label="raw")
            ax_bot.plot(t_sel, y_sm,  lw=1.2, color="steelblue",      label="smooth")
            ax_bot.axhline(TRIGGER_V, color="red", lw=0.8, ls="--")
            ax_bot.plot(t_sel[peaks], y_sm[peaks], "rv", ms=8, label=f"maxima n={n}")
            ax_bot.set(title=f"[{sel['xmin']:.3e} – {sel['xmax']:.3e} s]  |  pieken: {n}",
                       xlabel="Tijd (s)", ylabel="V")
            ax_bot.legend(fontsize=8)
            info.set_text(f"② Pieken geteld: {n}  —  druk Enter om te bevestigen, "
                          f"of sleep opnieuw voor nieuwe selectie.")
            info.set_color("darkorange")

            resultaat["n"]       = n
            resultaat["xmin"]    = sel["xmin"]
            resultaat["xmax"]    = sel["xmax"]
            resultaat["samples"] = int(mask.sum())
            staat["fase"]        = "check"
            fig.canvas.draw_idle()

        elif staat["fase"] == "check":
            resultaat["bevestigd"] = True
            staat["fase"]          = "klaar"
            info.set_text("✓ Bevestigd — venster sluit...")
            info.set_color("darkgreen")
            fig.canvas.draw_idle()
            plt.close(fig)

    def on_select_reset(xmin, xmax):
        if staat["fase"] == "check":
            staat["fase"] = "selecteer"
            info.set_color("darkred")
        on_select(xmin, xmax)

    span = widgets.SpanSelector(ax_top, on_select_reset, "horizontal", useblit=True, interactive=True,
                                drag_from_anywhere=True, props=dict(alpha=0.3, facecolor="orange"))
    fig.canvas.mpl_connect("key_press_event", on_enter)
    plt.show()

    if not resultaat["bevestigd"]:
        print("Niet bevestigd, meting overgeslagen.")
        return

    n_auto = resultaat["n"]
    print(f"\nBereik : {resultaat['xmin']:.4e} – {resultaat['xmax']:.4e} s"
          f"  |  samples: {resultaat['samples']}  |  pieken (auto): {n_auto}")

    raw = input(f"Aantal pieken [{n_auto}]  (Enter = overnemen): ").strip()
    n_stap = int(raw) if raw else n_auto

    dN_cum += n_stap
    print(f"Cumulatief ΔN = {dN_cum}")

    p = float(input("Druk p op dit moment (kPa): "))
    metingen.append((p, dN_cum))
    print(f"Opgeslagen: p={p} kPa, ΔN={dN_cum}  |  Totaal meetpunten: {len(metingen)}")


# Fit en grafieken 
def run_fit():
    if p_atm is None or len(metingen) < 2:
        print("Minimaal nulpunt + 2 metingen nodig.")
        return

    p_arr  = np.array([m[0] for m in metingen])       # kPa
    dN_arr = np.array([m[1] for m in metingen])        # cumulatief

    dp_arr = (p_atm - p_arr) * 1e3                    # Pa, positief bij dalende druk

    # Stap 3: fit |ΔN| = a · delfta p  (door oorsprong)
    def model_lin(dp, a):
        return a * dp

    popt, pcov = curve_fit(model_lin, dp_arr, dN_arr)
    a      = popt[0]
    a_err  = np.sqrt(pcov[0, 0])

    #  Stap 4: n(p_atm) 
    n_atm     = 1 + (a * LAMBDA * p_atm * 1e3) / (2 * L)
    n_atm_err = (a_err / a) * (n_atm - 1)

    # R²
    dN_pred = model_lin(dp_arr, a)
    ss_res  = np.sum((dN_arr - dN_pred)**2)
    ss_tot  = np.sum((dN_arr - dN_arr.mean())**2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    print(f"\n{'─'*40}")
    print(f"Meetpunten      : {len(metingen)}")
    print(f"Helling a       : {a:.4e} ± {a_err:.2e} franjes/Pa")
    print(f"R²              : {r2:.4f}")
    print(f"n(p_atm)        : {n_atm:.6f} ± {n_atm_err:.6f}")
    print(f"Referentie lucht: 1.000293")
    print(f"{'─'*40}")

    # Stap 5: n(p) per meetpunt 
    n_punten = 1 + (n_atm - 1) * p_arr / p_atm

    # Stap 3 grafiek: delfta p over delta N 
    dp_fit  = np.linspace(0, dp_arr.max(), 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(dp_arr / 1e3, dN_arr, color="steelblue", zorder=3, label="meetpunten")
    ax1.plot(dp_fit / 1e3, model_lin(dp_fit, a), color="red",
             label=f"fit (door oorsprong)\na = {a:.3e} franjes/Pa")
    ax1.set_xlabel("Drukdaling $p_{atm} - p$ (kPa)")
    ax1.set_ylabel("|ΔN| (franjes)")
    ax1.set_title("Franjes vs. drukdaling")
    ax1.legend()

    # Stap 6 grafiek: p over n(p) 
    p_fit   = np.linspace(0, p_atm, 300)
    n_fit   = 1 + (n_atm - 1) * p_fit / p_atm

    ax2.scatter(p_arr, n_punten, color="steelblue", zorder=3, label="meetpunten")
    ax2.plot(p_fit, n_fit, color="red",
             label=f"n(p) = 1 + (n_atm−1)·p/p_atm\nn_atm = {n_atm:.6f}")
    ax2.set_xlabel("Druk p (kPa)")
    ax2.set_ylabel("Brekingsindex n")
    ax2.set_title("Brekingsindex vs. druk")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Main
def main():
    while True:
        status = f"p_atm={p_atm} kPa  |  ΔN_cum={dN_cum}  |  meetpunten={len(metingen)}"
        print(f"\n{'─'*40}")
        print(status)
        print("0  Nulpunt instellen (begin experiment)")
        print("1  Nieuwe CSV — pieken tellen")
        print("2  Grafieken + fit")
        print("q  Afsluiten")
        keuze = input("Keuze: ").strip().lower()

        if   keuze == "0": run_nulpunt()
        elif keuze == "1": run_oscilloscoop()
        elif keuze == "2": run_fit()
        elif keuze == "q": break
        else: print("Ongeldige keuze.")

main()
