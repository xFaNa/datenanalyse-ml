import pandas as pd

# Aufgabenteil a)
df = pd.read_csv(
    "bundeslaender.txt",
    sep= r"\s+",    # Trennung nach Leerzeichen
)

df = df[["land", "area", "female", "male"]] # Vertauschen von Female und Male Column

df["population"] = df["male"] + df["female"]            # Berechnung der Gesamtbevölkerung (Nicht in realer Zahl angezeigt)
df["density"] = (df["population"] * 1000) / df["area"]  # Population * 1000 um echte Anzahl darzustellen für die Berechnung
df["density"] = df['density'].round(0)                  # Density auf eine Nachkommastelle runden
print("Aufgabenteil a - Population & Density")
print(df)

# Aufgabenteil b)
female_bigger_than_male = df[df["female"] > df["male"]]
print("\nAufgabeteil b - weibliche Bevölkerungsanzahl größer als männliche")
print("Anzahl: ", len(female_bigger_than_male))

# Aufgabenteil c)
density_gt_1000 = df[df["density"] > 1000]
print("\nAufgabenteil c - Bevölkerungsdichte größer als 1000")
print(density_gt_1000)