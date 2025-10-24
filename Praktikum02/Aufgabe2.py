import pandas as pd

# Aufgabenteil a)
personendaten = {
    "Gewicht": [65, 58, 58, 45, 43, 99, 68, 60],
    "Größe": [179, 165, 172, 154, 150, 189, 176, 175]
}

# Index setzen (ersetzt den generierten Index)
index = ["Henry", "Sarah", "Elke", "Lulu", "Vera", "Toni", "Maria", "Chris"]

df = pd.DataFrame(personendaten, index=index)
print("DataFrame (a):")
print(df)

# Aufgabenteil b)
bmi = df["Gewicht"] / (df["Größe"] / 100)**2
normalgewicht = df[(bmi >= 18.5) & (bmi <= 25)]
print("\nDataFrane (b) - Normalgewicht:")
print(normalgewicht)

# Aufgabenteil c)
namen_mit_e = df[df.index.str.contains("e")]
print("\nDataFrame (c) - Namen mit e enthalten:")
print(namen_mit_e)

# Aufgabenteil d)
df["BMI"] = df["Gewicht"] / (df["Größe"] / 100)**2
print("\nDataFrame (d) - mit BMI")
print(df)

# Aufgabenteil e)
untergewicht_mit_e = df[(bmi <= 20) & (df.index.str.contains("e"))]
print("\nDataFrame (e) - BMI unter 20 und Name mit e:")
print(untergewicht_mit_e)