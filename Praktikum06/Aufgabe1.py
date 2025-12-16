import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)          # alle Spalten anzeigen
pd.set_option("display.expand_frame_repr", False)   # Zeilen nicht umbrechen

# CSV laden
df = pd.read_csv("herford_weather.csv")

# Zeit in datetime umwandeln
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Als Index setzen
df = df.set_index("time")

# Erste 5 Zeilen und 5 Spalten anzeigen
print(df.iloc[:5, :5])

plt.figure(figsize=(12, 5))
df["temperature_2m (°C)"].plot()
plt.title("Temperatur in 2m Höhe – Herford")
plt.xlabel("Zeit")
plt.ylabel("Temperatur (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()
