import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")
variety = ["Setosa", "Versicolor", "Virginica"]
colors = ["red", "green", "blue"]

# Aufgabenteil a)
# Scatterplott für Kelchblatt
plt.figure(figsize=(6, 4))

for var, color in zip(variety, colors):
    daten = df[df["variety"] == var]
    plt.scatter(
        daten["sepal.length"],
        daten["sepal.width"],
        color=color,
        label=var,
        alpha=1
    )

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Scatterplott für Blütenblatt

plt.figure(figsize=(6, 4))

for var, color in zip(variety, colors):
    daten = df[df["variety"] == var]
    plt.scatter(
        daten["petal.length"],
        daten["petal.width"],
        color=color,
        label=var,
        alpha= 1
    )

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Aufgabenteil b)
# Kelchblatt Korrelationskoeffizient
corr_sepal = df["sepal.length"].corr(df["sepal.width"])

# Blütenblatt Korrelationskoeffizient
corr_petal = df["petal.length"].corr(df["petal.width"])

print(f"Der Korrelationskoeffizient zwischen Kelchblattlänge (cm) und Kelchblattbreite (cm) ist {corr_sepal:.5f}.")
print(f"Der Korrelationskoeffizient zwischen Blütenblattlänge (cm) und Blütenblattbreite (cm) ist {corr_petal:.5f}.")