# ============================================================
# Praktikum 07 – Aufgabe 1f
# Lineare Regression zur Schätzung von Schneefall (snowfall (cm))
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras

sns.set_style("whitegrid")

# ------------------------------------------------------------
# 1) Datensatz laden
# ------------------------------------------------------------
df = pd.read_csv("herford_weather.csv")
print("Datensatzgröße:", df.shape)

# ------------------------------------------------------------
# 2) Spalten selektieren (wie in Aufgabe a vorgegeben)
# ------------------------------------------------------------
cols = [
    'temperature_2m (°C)', 'relativehumidity_2m (%)', 'dewpoint_2m (°C)',
    'apparent_temperature (°C)', 'pressure_msl (hPa)', 'surface_pressure (hPa)',
    'precipitation (mm)', 'rain (mm)', 'snowfall (cm)', 'weathercode (wmo code)',
    'cloudcover (%)', 'cloudcover_low (%)', 'cloudcover_mid (%)', 'cloudcover_high (%)',
    'shortwave_radiation (W/m²)', 'direct_radiation (W/m²)', 'diffuse_radiation (W/m²)',
    'direct_normal_irradiance (W/m²)', 'windspeed_10m (km/h)', 'windspeed_100m (km/h)',
    'winddirection_10m (°)', 'winddirection_100m (°)', 'windgusts_10m (km/h)',
    'et0_fao_evapotranspiration (mm)', 'vapor_pressure_deficit (kPa)',
    'soil_temperature_0_to_7cm (°C)', 'soil_temperature_7_to_28cm (°C)',
    'soil_temperature_28_to_100cm (°C)', 'soil_temperature_100_to_255cm (°C)',
    'soil_moisture_0_to_7cm (m³/m³)', 'soil_moisture_7_to_28cm (m³/m³)',
    'soil_moisture_28_to_100cm (m³/m³)', 'soil_moisture_100_to_255cm (m³/m³)'
]

df_sel = df[cols].copy()
print("NaN-Werte gesamt:", df_sel.isna().sum().sum())

# ------------------------------------------------------------
# 3) Feature-Auswahl über Korrelation mit Schneefall
# ------------------------------------------------------------
target = "snowfall (cm)"

corr = df_sel.corr(numeric_only=True)
corr_with_snow = corr[target].sort_values(ascending=False)

print("\nKorrelationen mit Schneefall:")
print(corr_with_snow)

# Top-5 Features nach absoluter Korrelation (ohne Target)
corr_abs = corr[target].abs().sort_values(ascending=False)
features = [c for c in corr_abs.index if c != target][:5]

print("\nGewählte Features für Schneefall:")
print(features)

# ------------------------------------------------------------
# 4) X / y bauen, Split & StandardScaler
# ------------------------------------------------------------
X = df_sel[features].values
y = df_sel[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nShapes:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)

# ------------------------------------------------------------
# 5) epochs & batch_size testen (kleiner sinnvoller Vergleich)
# ------------------------------------------------------------
configs = [
    {"epochs": 10, "batch_size": 128},
    {"epochs": 20, "batch_size": 32},
    {"epochs": 20, "batch_size": 256}
]

results = []  # (cfg, val_loss, history, model)

for cfg in configs:
    print("\nTraining mit:", cfg)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=0
    )

    val_loss = history.history["val_loss"][-1]
    results.append((cfg, val_loss, history, model))

    print("Finaler val_loss:", val_loss)

# ------------------------------------------------------------
# 6) Bestes Modell auswählen
# ------------------------------------------------------------
best_cfg, best_val_loss, best_history, best_model = min(results, key=lambda x: x[1])

print("\nBeste Konfiguration:", best_cfg)
print("Bester finaler val_loss:", best_val_loss)

# ------------------------------------------------------------
# 7) R² auf Testset berechnen
# ------------------------------------------------------------
y_pred = best_model.predict(X_test_scaled).flatten()
r2 = r2_score(y_test, y_pred)

print("\nR² Wert (Schneefall):", r2)

# ------------------------------------------------------------
# 8) Lernkurve plotten
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(best_history.history["loss"], label="Trainingsverlust (MSE)")
plt.plot(best_history.history["val_loss"], label="Validierungsverlust (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Verlust (MSE)")
plt.title(
    f"Lernkurve – Schneefall "
    f"(epochs={best_cfg['epochs']}, batch_size={best_cfg['batch_size']})"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9) Interpretation (kurzer Hinweis)
# ------------------------------------------------------------
print("\nInterpretation:")
print(
    "Der R²-Wert ist gering bzw. negativ. "
    "Schneefall ist stark ereignis- und schwellenbasiert "
    "(z.B. Temperatur um 0°C + Niederschlag) "
    "und lässt sich daher mit einer linearen Regression nur schlecht modellieren."
)
