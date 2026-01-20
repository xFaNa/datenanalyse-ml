import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# Einstellungen
# =========================
CSV_PATH = "pvtest.csv"
FEATURES = ["Edaily", "Dci", "Dcp", "Dcu", "Temp1", "hour"]
TARGET = "Dcp"

TRAIN_RATIO = 0.8

WINDOW = 36      # 3 Stunden bei 5-Minuten Raster -> 36 Schritte
HORIZON = 1      # 1 Schritt in die Zukunft -> +5 Minuten


# =========================
# a) Zeitreihe bauen + Ausschnitt 16:25–17:40 wie Aufgabenbild ausgeben
# =========================
df = pd.read_csv(CSV_PATH)

df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"]).copy()

df["hour"] = df["Time"].dt.hour
df = df[["Time"] + FEATURES].dropna()

df = df.set_index("Time").sort_index()
df.index.name = "Time"

# 5-Minuten Raster erzwingen + Lücken füllen
df = df.asfreq("5min")
df[FEATURES] = df[FEATURES].interpolate(method="time").ffill().bfill()

# --- Ausgabe exakt im Stil des Aufgabenbildes (16:25–17:40) ---
start_t = "16:25"
end_t = "17:40"
target_day = pd.Timestamp("2021-01-26")

# Versuch 1: exakt wie Screenshot-Tag
slice_df = df.loc[target_day:target_day + pd.Timedelta(days=1)].between_time(start_t, end_t)

# Fallback: erster Tag, der dieses Uhrzeitfenster enthält
if slice_df.empty:
    candidates = df.between_time(start_t, end_t)
    if candidates.empty:
        raise ValueError(f"Kein Datensatz im Zeitfenster {start_t}–{end_t} gefunden.")
    first_day = candidates.index.normalize().unique()[0]
    slice_df = df.loc[first_day:first_day + pd.Timedelta(days=1)].between_time(start_t, end_t)

print("a) Zeitreihen-DataFrame (Ausschnitt 16:25–17:40 wie Aufgabenbild):")
print(slice_df[FEATURES].round(6))


# =========================
# b) Train/Test Split (chronologisch)
# =========================
split_idx = int(len(df) * TRAIN_RATIO)
df_train = df.iloc[:split_idx]
df_test  = df.iloc[split_idx:]

print(f"\nb) Train: {len(df_train)}, Test: {len(df_test)}")


# =========================
# c) Standardisierung + Sliding Windows
# =========================
X_train_raw = df_train[FEATURES].values
y_train_raw = df_train[[TARGET]].values  # 2D für scaler_y

X_test_raw = df_test[FEATURES].values
y_test_raw = df_test[[TARGET]].values

# X und y getrennt skalieren (klarer + sauber rücktransformierbar)
scaler_x = StandardScaler()
X_train_std = scaler_x.fit_transform(X_train_raw)
X_test_std  = scaler_x.transform(X_test_raw)

scaler_y = StandardScaler()
y_train_std = scaler_y.fit_transform(y_train_raw)
y_test_std  = scaler_y.transform(y_test_raw)

def restructure_data(X, y, window, horizon):
    X_, y_ = [], []
    for idx in range(len(X) - (window + horizon)):
        X_.append(X[idx:idx + window])
        y_.append(y[idx + window + horizon])  # Zielwert in der Zukunft
    return np.array(X_, dtype=np.float32), np.array(y_, dtype=np.float32)

X_train, y_train = restructure_data(X_train_std, y_train_std, WINDOW, HORIZON)
X_test, y_test   = restructure_data(X_test_std,  y_test_std,  WINDOW, HORIZON)

print(f"\nc) X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test : {X_test.shape},  y_test : {y_test.shape}")


# =========================
# d) LSTM + Dropout Modell
# =========================
model = Sequential([
    Input(shape=(WINDOW, len(FEATURES))),
    LSTM(64, return_sequences=True, dropout=0.2),
    LSTM(32, dropout=0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("\nd) Model Summary:")
model.summary()


# =========================
# e) Training (>=100 Epochen + EarlyStopping)
# =========================
early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,     # Zeitreihe nicht shufflen!
    callbacks=[early],
    verbose=1
)

# Vorhersage (skaliert)
y_pred_std = model.predict(X_test, verbose=0)

# Rücktransformation in Originalskala
y_pred = scaler_y.inverse_transform(y_pred_std).flatten()

# y_true passend zu X_test: window+horizon Schritte vorne weg
y_true = y_test_raw[WINDOW + HORIZON:].flatten()

mae = mean_absolute_error(y_true, y_pred)
print(f"\ne) Test MAE (original Dcp): {mae:.2f}")


# =========================
# f) EIN Plot wie im Aufgabenblatt (Index 0..300, aktiver Abschnitt)
# =========================
peak = int(np.argmax(y_true))
start = max(0, peak - 150)
end   = min(len(y_true), peak + 150)

yt = y_true[start:end]
yp = y_pred[start:end]
x = np.arange(len(yt))

plt.figure(figsize=(12, 5))
plt.plot(x, yt, label="Gemessen")
plt.plot(x, yp, label="Vorhergesagt")
plt.xlabel("Zeitindex")
plt.ylabel("Dcp")
plt.title("LSTM Prognose: Dcp (+5 Minuten) – Gemessen vs. Vorhergesagt")
plt.legend()
plt.tight_layout()
plt.show()
