import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
variety = ["Setosa", "Versicolor", "Virginica"]
colors = ["blue", "red", "green"]

for feature in features:
    plt.figure(figsize=(6, 4))

    for var, color in zip(variety, colors):
        daten = df[df["variety"] == var][feature]
        plt.hist(daten, bins=10, alpha=0.5, color=color, label=var)

    plt.xlabel(feature.replace(".", " ") + " (cm)")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.show()
