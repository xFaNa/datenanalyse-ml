import matplotlib.pyplot as plt

# ---- Teil a Säulendiagramm ------
countries = ["Asien", "Afrika", "N-Amerika", "S-Amerika", "Antarktis", "Europa", "Australien"]
country_sizes = [44.4, 30.3, 24.9, 17.8, 13.2, 10.5, 8.5]

# Säulendiagramm zeichnen
plt.figure(figsize=(8, 6))
plt.bar(countries, country_sizes, color="green", edgecolor="black")

# Beschriftung
plt.xlabel("Kontinente")
plt.ylabel("Größe in Mio. km2")
plt.title("a) Flächen der Kontinente (Säulendiagramm)")
plt.grid(axis="y", linestyle="--")

#Diagramm anzeigen
plt.show()

# === Teil b: Tortendiagramm ===
total_earth = 510.0
land_total = sum(country_sizes)
water = total_earth - land_total

labels = countries + ["Wasser"]
values = country_sizes + [water]
colors = ["yellow", "brown", "purple", "orange", "cyan", "red", "green", "blue"]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("b) Verteilung der Erdoberfläche (Kontinente & Wasser)")
plt.show()