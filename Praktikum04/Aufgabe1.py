import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

# Listen für den Schleifendurchlauf
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
variety = ["Setosa", "Versicolor", "Virginica"]
colors = ["blue", "red", "green"]

# Iteriere durch features List um von jedem ein Histogramm zu erzeugen
for feature in features:
    plt.figure(figsize=(6, 4)) # Bildgröße
    # Innere Schleife für die drei varieties
    for var, color in zip(variety, colors):    # verbindet Listen paarweise
        daten = df[df["variety"] == var][feature]
        plt.hist(daten, bins=10, alpha=0.5, color=color, label=var)

    plt.xlabel(feature.replace(".", " ") + " (cm)")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.show()
