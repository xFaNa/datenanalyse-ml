import pandas as pd
import matplotlib.pyplot as plt

# Allgemeine Einstellungen für die Anzeige (nur für Konsole)
pd.set_option("display.max_columns", None)          # alle Spalten anzeigen
pd.set_option("display.expand_frame_repr", False)   # Zeilen nicht umbrechen


# Datei einlesen
# index_col=0: die erste Spalte (232, 233, ...) als Index benutzen
df = pd.read_csv("pvtest.csv", index_col=0)


# Aufgabe 3a)
# Erste 10 Zeilen des gesamten DataFrames anzeigen
print("Aufgabe 3a – erste 10 Zeilen des DataFrames:\n")
print(df.head(10))


# Aufgabenteil b
# Spaltennamen anzeigen und nur Dci, Dcp, Dcu, Temp1 selektieren
print("\nAufgabe 3b – Spaltennamen:\n")
print(df.columns)

# nur die vier interessierenden Merkmale auswählen
df_four = df[["Dci", "Dcp", "Dcu", "Temp1"]]

print("\nAufgabe 3b – Dci, Dcp, Dcu, Temp1 (erste 10 Zeilen):\n")
print(df_four.head(10))


# Aufgabenteil c
# Alle Zeilen herausfiltern, in denen Dci, Dcp, Dcu ODER Temp1 = 0 ist.
# Übrig bleiben nur Zeilen, in denen ALLE vier > 0 sind.
mask_non_zero = (
    (df_four["Dci"]   != 0) &
    (df_four["Dcp"]   != 0) &
    (df_four["Dcu"]   != 0) &
    (df_four["Temp1"] != 0)
)

df_clean = df_four[mask_non_zero]

print("\nAufgabe 3c – Zeilen ohne Nullwerte in Dci/Dcp/Dcu/Temp1 (erste 10 Zeilen):\n")
print(df_clean.head(10))


# Aufgabenteil d
corr = df_clean.corr()   # Korrelationsmatrix berechnen
print("\nKorrelationsmatrix von Dci, Dcp, Dcu, Temp1:\n")
print(corr)

# Heatmap zeichnen
fig, ax = plt.subplots(figsize=(5, 4))

# Matrix als Bild darstellen
im = ax.imshow(corr, origin="upper")

# Achsenbeschriftungen setzen
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)

# Korrelationswerte in die Zellen schreiben
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        ax.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center", va="center", color="white"
        )

# Farbskala rechts
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
