import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter, medfilt

bestand = Path("Project/team_11_brekingsindex_bepalen/Newfile15.csv")

# Minimale afstand tussen twee afzonderlijke franjes, in meetpunten.
# Start bijvoorbeeld met 5 tot 8.
distance = 10

# Relatieve minimale prominence.
# Start bijvoorbeeld met 0.05 tot 0.10.
prominence_factor = 0.08

parameters = pd.read_csv(bestand, nrows=1)

start = float(parameters.loc[0, "Start"])
increment = float(parameters.loc[0, "Increment"])

df = pd.read_csv(
    bestand,
    skiprows=[1],
    usecols=["X", "CH1"]
)

df["X"] = pd.to_numeric(df["X"])
df["CH1"] = pd.to_numeric(df["CH1"])

df["tijd_s"] = start + df["X"] * increment

tijd = df["tijd_s"].to_numpy()
spanning = df["CH1"].to_numpy()

# SIGNAAL BEWERKEN
# Verwijdert losse hoge of lage uitschieters.
# Deze instelling blijft voor alle bestanden gelijk.
spanning_schoon = medfilt(spanning, kernel_size=3)

# Het gladstrijkvenster wordt gekoppeld aan de ingestelde distance.
# Het blijft oneven en is minimaal 5 meetpunten breed.
window_length = max(5, distance if distance % 2 == 1 else distance + 1)

# Zorg dat het venster altijd kleiner is dan het aantal meetpunten.
if window_length >= len(spanning_schoon):
    window_length = len(spanning_schoon) - 1

if window_length % 2 == 0:
    window_length -= 1

gladde_spanning = savgol_filter(
    spanning_schoon,
    window_length=window_length,
    polyorder=2
)

# PIEKDETECTIE

spanningsbereik = gladde_spanning.max() - gladde_spanning.min()
prominence = prominence_factor * spanningsbereik

pieken, eigenschappen = find_peaks(
    gladde_spanning,
    prominence=prominence,
    distance=distance
)

# RESULTATEN
dt = tijd[1] - tijd[0]
samplefrequentie = 1 / dt
minimale_afstand_ms = distance * dt * 1000

print(f"Bestand: {bestand.name}")
print(f"Tijdstap: {dt:.6f} s")
print(f"Samplefrequentie: {samplefrequentie:.1f} Hz")
print(f"Gladstrijkvenster: {window_length} meetpunten")
print(f"Minimale piekafstand: {distance} meetpunten = {minimale_afstand_ms:.2f} ms")
print(f"Spanningsbereik: {spanningsbereik:.4f} V")
print(f"Prominence: {prominence:.4f} V ({prominence_factor:.0%} van bereik)")
print(f"Aantal gevonden pieken: {len(pieken)}")

# Sla de posities van de gedetecteerde pieken op
df_pieken = pd.DataFrame({
    "piek_nummer": range(1, len(pieken) + 1),
    "index": pieken,
    "tijd_s": tijd[pieken],
    "spanning_V": spanning[pieken],
    "gladde_spanning_V": gladde_spanning[pieken]
})

uitvoer_csv = bestand.with_name(f"{bestand.stem}_pieken.csv")
df_pieken.to_csv(uitvoer_csv, index=False)


# GRAFIEK
plt.figure(figsize=(14, 6))

plt.plot(
    tijd,
    spanning,
    color="steelblue",
    alpha=0.30,
    linewidth=0.7,
    label="Origineel signaal"
)

plt.plot(
    tijd,
    gladde_spanning,
    color="black",
    linewidth=1.2,
    label="Gefilterd signaal"
)

plt.plot(
    tijd[pieken],
    gladde_spanning[pieken],
    "rx",
    markersize=8,
    markeredgewidth=2,
    label=f"Gevonden pieken ({len(pieken)})"
)

plt.xlabel("Tijd (s)")
plt.ylabel("Spanning (V)")
plt.title(f"Interferentiesignaal met gedetecteerde pieken — {bestand.stem}")
plt.grid()
plt.legend()
plt.tight_layout()

uitvoer_grafiek = bestand.with_name(f"{bestand.stem}_pieken.png")
plt.savefig(uitvoer_grafiek, dpi=300)
plt.show()

print(f"Grafiek opgeslagen als: {uitvoer_grafiek}")
print(f"Piekgegevens opgeslagen als: {uitvoer_csv}")
