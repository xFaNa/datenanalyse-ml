import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ========== a) DataFrame erstellen ==========
df = pd.read_csv("pvtest.csv")
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"])

df["hour"] = df["Time"].dt.hour
df = df[["Time", "Edaily", "Dci", "Dcp", "Dcu", "Temp1", "hour"]].dropna()
df = df.set_index("Time").sort_index()

# (optional aber sauber) 5-Minuten Raster erzwingen
df = df.asfreq("5min")
df = df.interpolate(method="time").ffill().bfill()

print("a) DataFrame:")
print(df.head(10))

# ========== b) Train/Test Split ==========
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df  = df.iloc[split_index:]

print(f"\nb) Train: {len(train_df)}, Test: {len(test_df)}")

# ========== c) Standardisierung + Datenfenster ==========
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df.values)
test_scaled  = scaler.transform(test_df.values)

window_size = 36
dcp_index = list(train_df.columns).index("Dcp")

def create_sequences(data, window_size, y_index):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, y_index])  # Dcp +5min (nächster Zeitschritt)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = create_sequences(train_scaled, window_size, dcp_index)

# Test "warm starten": letzte window_size aus Train vor Test hängen
test_warm = np.vstack([train_scaled[-window_size:], test_scaled])
X_test, y_test = create_sequences(test_warm, window_size, dcp_index)

print(f"\nc) X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# ========== d) LSTM-Modell definieren ==========
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ========== e) Training (zeitliche Validation, kein Shuffle) ==========
val_split = 0.2
val_start = int(len(X_train) * (1 - val_split))
X_tr, y_tr = X_train[:val_start], y_train[:val_start]
X_val, y_val = X_train[val_start:], y_train[val_start:]

early_stop = EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-5)

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=120,          # >=100
    batch_size=32,
    shuffle=False,       # wichtig für Zeitreihen
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test, verbose=0)

# MAE zurück in Originalskala (für Schwelle wie 50 relevant!)
mae_original = mae * scaler.scale_[dcp_index]
print(f"\ne) Test MAE (skaliert): {mae:.4f}")
print(f"   Test MAE (original Dcp): {mae_original:.2f}")

# ========== f) Visualisierung ==========
y_pred = model.predict(X_test, verbose=0).flatten()

y_test_original = y_test * scaler.scale_[dcp_index] + scaler.mean_[dcp_index]
y_pred_original = y_pred * scaler.scale_[dcp_index] + scaler.mean_[dcp_index]

plt.figure(figsize=(12, 5))
plt.plot(y_test_original[:300], label="Gemessen")
plt.plot(y_pred_original[:300], label="Vorhergesagt")
plt.xlabel("Zeitschritte")
plt.ylabel("Dcp")
plt.title("LSTM Prognose: Dcp (+5 Minuten) – Gemessen vs. Vorhergesagt")
plt.legend()
plt.tight_layout()
plt.show()
