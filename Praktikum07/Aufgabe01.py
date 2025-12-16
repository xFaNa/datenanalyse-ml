import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score

sns.set_style("whitegrid")

# CSV Datei laden
df = pd.read_csv("herford_weather.csv")
print("Datensatzgröße:", df.shape)

# Aufgabenteil a) Spalten selektieren
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
df_selected = df[cols].copy()

# Checke ob Werte fehlen
nan_counts = df_selected.isna().sum().sum()
print("Gesamte NaNs in ausgewählten Spalten:", nan_counts)

corr = df_selected.corr(numeric_only=True)

# Heatmap plotten
plt.figure(figsize=(12,9))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Korrelation Heatmap")
plt.tight_layout()
plt.show()

# Korrelation in Bezug auf Taupunkt berechnen und ausgeben
corr_with_dewpoint = corr["dewpoint_2m (°C)"].sort_values(ascending=False)
print("\nKorrelation mit Taupunkt 2m (°C)")
print(corr_with_dewpoint)

# Ein Scatterplot zur linearen Plausibilitätsprüfung
plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=df_selected['temperature_2m (°C)'],
    y=df_selected['dewpoint_2m (°C)'],
    s=5, alpha=0.25
)
plt.title("Temperature (2m) vs Dewpoint")
plt.tight_layout()
plt.show()

# Feature-Auswahl für Taupunkt
features_dew = [
    "temperature_2m (°C)",
    "relativehumidity_2m (%)",
    "soil_temperature_0_to_7cm (°C)"
]
target_dew = "dewpoint_2m (°C)"

print("\nSelected features for dewpoint prediction:")
print("Features:", features_dew)
print("Target:", target_dew)

# Aufgabenteil b) StandardScaler und Keras Modell

# X/y bauen
X = df_selected[features_dew].values
y = df_selected[target_dew].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nShapes after scaling:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)

# Lineares Keras-Modell bauen
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(1, activation="linear")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

print("\nModell Zusammenfassung:")
model.summary()

# Trainieren
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=128,
    verbose=1
)

# Aufgabenteil c) epochs und batch-sizes testen
configs = [
    {"epochs": 10, "batch-size": 128},
    {"epochs": 20, "batch-size": 32},
    {"epochs": 20, "batch-size": 256}
]

results = [] # speicher für cfg, final_val_loss, history, model

for cfg in configs:
    print("\nTraining mit:", cfg)

    tmp_model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(1, activation="linear")
    ])

    tmp_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    tmp_history = tmp_model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=cfg["epochs"],
        batch_size=cfg["batch-size"],
        verbose=0
    )

    final_val_loss = tmp_history.history["val_loss"][-1]
    results.append((cfg, final_val_loss, tmp_history, tmp_model))

    print("Finaler val_loss:", final_val_loss)

best_cfg, best_val_loss, best_history, best_model = min(results, key=lambda x: x[1])

print("\nBeste Konfiguration:", best_cfg)
print("Bester finaler val_loss:", best_val_loss)


# d) R^2 für bestes Modell
y_predict = best_model.predict(X_test_scaled).flatten()
r2 = r2_score(y_test, y_predict)
print("\nR² Wert (bestes Modell):", r2)


# e) Lernkurve für bestes Ergebnis plotten
plt.figure(figsize=(8, 5))
plt.plot(best_history.history["loss"], label="Trainingsverlust (MSE)")
plt.plot(best_history.history["val_loss"], label="Validierungsverlust (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Verlust (MSE)")
plt.title(f"Lernkurve – bestes Modell (epochs={best_cfg['epochs']}, batch_size={best_cfg['batch-size']})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()