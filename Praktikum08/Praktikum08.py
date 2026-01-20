import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# 0) Daten vorbereiten
# -------------------------
df = pd.read_csv("rawdata_luftqualitaet.csv")

FEATURES = [
    "humidity_inside",
    "temperature_inside",
    "co2_inside",
    "temperature_heater",
    "temperature_wall_inside",
]
LABEL = "state_air_quality"

X = df[FEATURES].to_numpy()
y = df[LABEL].to_numpy()

# gleicher Train/Test Split für alle Aufgaben (wichtig!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# StandardScaler: fit nur auf Train (kein Data Leakage!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def build_base_model(input_dim: int, l2_lambda: float | None = None) -> keras.Model:
    reg = keras.regularizers.l2(l2_lambda) if l2_lambda is not None else None

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(60, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(60, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_loss(history, title: str):
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="test loss")
    plt.xlabel("epochs")
    plt.ylabel("loss (sparse cross entropy)")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()


# =====================================================
# Aufgabe 1: Basismodell (ohne Callbacks)
# =====================================================
print("\n" + "=" * 60)
print("AUFGABE 1 – MODELLSUMMARY")
print("=" * 60)

model1 = build_base_model(input_dim=5)
model1.summary()

history1 = model1.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

plot_loss(history1, "Aufgabe 1: Basismodell (Overfitting sichtbar)")


# =====================================================
# Aufgabe 2: Basismodell + EarlyStopping
# =====================================================
print("\n" + "=" * 60)
print("AUFGABE 2 – MODELLSUMMARY (EarlyStopping)")
print("=" * 60)

model2 = build_base_model(input_dim=5)
model2.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history2 = model2.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

plot_loss(history2, "Aufgabe 2: EarlyStopping (Overfitting reduziert)")


# =====================================================
# Aufgabe 3: Basismodell + L2-Regularisierung
# =====================================================
print("\n" + "=" * 60)
print("AUFGABE 3 – MODELLSUMMARY (L2-Regularisierung)")
print("=" * 60)

model3 = build_base_model(input_dim=5, l2_lambda=0.001)
model3.summary()

history3 = model3.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

plot_loss(history3, "Aufgabe 3: L2-Regularisierung (Overfitting reduziert)")
