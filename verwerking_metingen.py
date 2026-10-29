import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

LAMBDA = 530e-9  # m golflengte
L = 0.10         # m licht doorloopt de gascel tweemaal
P_ATM = 101.3    # kPa referentiedruk voor de brekingsindex

# [drukverlies in kPa, franjes tijdens dat drukverlies]. 28 metingen
metingen = [[27.963,25],[15.812,13],[28.690,24],[27.711,23],[29.100,28],[29.384,24],[29.321,24],[15.086,12],[14.866,12],[14.045,12],
            [11.899,9],[11.394,9],[11.836,10],[5.586,4],[5.532,4],[5.586,4],[17.517,15],[17.327,14],[15.402,13],[15.528,14],
            [13.761,12],[13.950,12],[13.635,11],[21.493,18],[22.914,20],[23.829,20],[24.334,21],[27.616,24]]

gegevens = np.asarray(metingen, dtype=float)
if gegevens.ndim != 2 or gegevens.shape[1] != 2:
    raise ValueError("Gebruik [drukverlies_kPa, aantal_franjes] per meetpunt.")
if len(gegevens) < 2:
    raise ValueError("Minimaal twee meetpunten nodig voor fit en onzekerheid.")
if not np.all(np.isfinite(gegevens)) or np.any(gegevens < 0):
    raise ValueError("Waarden moeten eindig en niet-negatief zijn.")

dp_kpa = gegevens[:, 0]
dp_pa = dp_kpa * 1e3
N = gegevens[:, 1]
if not np.any(dp_pa > 0):
    raise ValueError("Minstens één positief drukverlies nodig.")


def model(dp_pa, a):
    return a * dp_pa


# Een ongewogen fit op ALLE punten, zonder middelen of extra nulpunt.
popt, pcov = curve_fit(model, dp_pa, N)
a = float(popt[0])  # franjes/Pa
sigma_a = float(np.sqrt(pcov[0, 0]))
if a <= 0:
    raise ValueError("Een positieve helling voor N, drukverlies wordt verwacht.")

# N = (2L/lambda) Δn; bij constant dn/dP geldt Δn = (dn/dP) ΔP.
dndP = a * LAMBDA / (2 * L)
sigma_dndP = sigma_a * LAMBDA / (2 * L)
n_atm = 1 + dndP * P_ATM * 1e3
sigma_n_atm = sigma_dndP * P_ATM * 1e3

# Student-t: betrouwbaarheidsniveaus gelijk aan de normale 1σ- en 3σ-dekking.
# Bij een eindige steekproef zijn de factoren niet exact 1 en 3.
dof = len(N) - 1  # alle punten minus één geschatte parameter
levels = [
    ("3σ-equivalent", 0.997300204, "lightskyblue", 0.45),
    ("1σ-equivalent", 0.682689492, "royalblue", 0.35),
]
factors = {
    name: student_t.ppf((1 + level) / 2, dof)
    for name, level, _, _ in levels
}

print(f"Fit op alle {len(N)} punten: N = a ΔP")
print(f"a = {a:.4e} ± {sigma_a:.2e} franjes/Pa (standaardfout)")
print(f"dn/dP = {dndP:.4e} ± {sigma_dndP:.2e} Pa⁻¹ (standaardfout)")
print(f"n({P_ATM} kPa) = {n_atm:.7f} ± {sigma_n_atm:.7f} (standaardfout)")
print(f"Vrijheidsgraden: {dof}")
for name, level, _, _ in levels:
    factor = factors[name]
    width_a = factor * sigma_a
    width_n = factor * sigma_n_atm
    print(f"\n{name} BI ({100 * level:.4f}%), t-factor = {factor:.3f}")
    print(f"  a ∈ [{a - width_a:.4e}, {a + width_a:.4e}] franjes/Pa")
    print(f"  n({P_ATM} kPa) ∈ [{n_atm - width_n:.7f}, {n_atm + width_n:.7f}]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
dp_fit_kpa = np.linspace(0, dp_kpa.max() * 1.05, 300)
dp_fit_pa = dp_fit_kpa * 1e3
N_fit = model(dp_fit_pa, a)
sigma_N_fit = np.abs(dp_fit_pa) * sigma_a

for name, level, color, alpha in levels:
    width = factors[name] * sigma_N_fit
    ax1.fill_between(dp_fit_kpa, N_fit - width, N_fit + width,
                     color=color, alpha=alpha, label=f"{name} BI")
ax1.scatter(dp_kpa, N, color="steelblue", edgecolors="black",
            zorder=3, label="metingen")
ax1.plot(dp_fit_kpa, N_fit, color="red",
         label=f"fit: N = a·ΔP\na = {1000*a:.6f}  ± {1000*sigma_a:.6f} franjes/kPa")
ax1.set(xlabel="Drukverlies ΔP (kPa)", ylabel="Franjes tijdens drukverlies N",
        title="franjeverschuiving over druk")
ax1.legend(fontsize=8)

P_fit_kpa = np.linspace(0, P_ATM, 300)
n_fit = 1 + dndP * P_fit_kpa * 1e3
sigma_n_fit = sigma_dndP * P_fit_kpa * 1e3
for name, level, color, alpha in levels:
    width = factors[name] * sigma_n_fit
    ax2.fill_between(P_fit_kpa, n_fit - width, n_fit + width,
                     color=color, alpha=alpha, label=f"{name} BI")
ax2.plot(P_fit_kpa, n_fit, color="red",
         label=f"n({P_ATM} kPa) = {n_atm:.7f} ± {sigma_n_atm:.7f}")
ax2.set(xlabel="Absolute druk P (kPa)", ylabel="Berekende brekingsindex n",
        title="Brekingsindex over druk")
ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
ax2.legend(fontsize=8)
fig.tight_layout()
plt.show()