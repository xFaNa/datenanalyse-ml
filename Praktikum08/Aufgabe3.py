
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras

# Konfiguration
CSV_PATH = "rawdata_luftqualitaet.csv"

FEATURES = [
    "humidity_inside",
    "temperature_inside",
    "co2_inside",
    "temperature_heater",
    "temperature_wall_inside"
]

LABEL = "state_air_quality"

TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001

df = pd.read_csv(CSV_PATH)

# Features/Label trennen
X = df[FEATURES].copy()
y = df[LABEL].copy()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify = y
)

# Skalieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

L2_LAMBDA = 0.001
l2_reg = keras.regularizers.l2(L2_LAMBDA)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(60, activation="relu", kernel_regularizer=l2_reg),
    keras.layers.Dense(60, activation="relu", kernel_regularizer=l2_reg),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_split = VALIDATION_SPLIT,
    verbose = 1,
)

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure()
plt.plot(train_loss, label="train loss")
plt.plot(val_loss, label="test loss")
plt.xlabel("epochs")
plt.ylabel("loss (sparse cross entropy)")
plt.grid(True)
plt.legend()
plt.show()