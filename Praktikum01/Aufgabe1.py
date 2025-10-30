import matplotlib.pyplot as plt

# Beispiel-Daten
monate = ["Jan", "Feb", "Mae", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
temperatur_deutschland = [1.5, 6.6, 7.5, 10.1, 15.1, 16.8, 18.8, 19.9, 15.4, 11.0, 5.1, 3.0]
temperatur_australien = [27, 26, 23, 19, 15, 11, 11, 12, 15, 19, 22, 25]


# Diagramm erstellen
plt.plot(monate, temperatur_deutschland, label="Deutschland", color="blue", marker="")
plt.plot(monate, temperatur_australien, label="Australien", color="red", marker="")


plt.xlabel("Monate")
plt.ylabel("Temperatur in Grad °C")
plt.title("durchschnitlliche Monatstemperaturen")
plt.legend()
plt.grid(True)

# Diagramm anzeigen
plt.show()

