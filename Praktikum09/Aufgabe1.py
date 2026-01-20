import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ========== a) DataFrame erstellen ==========
df = pd.read_csv('pvtest.csv')
df['Time'] = pd.to_datetime(df['Time'])
df['hour'] = df['Time'].dt.hour
df = df[['Time', 'Edaily', 'Dci', 'Dcp', 'Dcu', 'Temp1', 'hour']]
df = df.set_index('Time')
df = df.dropna()

print("a) DataFrame:")
print(df.head(10))

# ========== b) Train/Test Split ==========
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print(f"\nb) Train: {len(train_df)}, Test: {len(test_df)}")

# ========== c) Standardisierung + Datenfenster ==========
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# 3 Stunden = 36 Zeitschritte (bei 5-Min-Intervallen)
window_size = 36
dcp_index = 2  # Index von Dcp in den Features

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, dcp_index])  # Dcp vorhersagen
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

print(f"\nc) X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")

# ========== d) LSTM-Modell definieren ==========
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, 6)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ========== e) Training ==========
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
loss, mae = model.evaluate(X_test, y_test, verbose=0)

# MAE zurück in Originalskala
mae_original = mae * scaler.scale_[dcp_index]
print(f"\ne) Test MAE (skaliert): {mae:.4f}")
print(f"   Test MAE (original): {mae_original:.2f}")

# ========== f) Visualisierung ==========
y_pred = model.predict(X_test)

# Rücktransformation in Originalskala
y_test_original = y_test * scaler.scale_[dcp_index] + scaler.mean_[dcp_index]
y_pred_original = y_pred.flatten() * scaler.scale_[dcp_index] + scaler.mean_[dcp_index]

plt.figure(figsize=(12, 5))
plt.plot(y_test_original[:300], label='Gemessen', color='blue')
plt.plot(y_pred_original[:300], label='Vorhergesagt', color='red')
plt.xlabel('Zeitschritte')
plt.ylabel('Dcp (Energieertrag)')
plt.title('LSTM Prognose: Gemessen vs. Vorhergesagt')
plt.legend()
plt.tight_layout()
plt.show()
