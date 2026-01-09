
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras


# Konfiguration
CSV_PATH = "rawdata_luftqualitaet.csv"

FEATURES = [
    "humidity_inside",
    "temperature_inside",
    "co2_inside",
    "temperature_heater",
    "temperature_wall_inside",
]
LABEL = "state_air_quality"

TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# Datenfunktionen
def load_data(path: str) -> pd.DataFrame:
    """Lädt den CSV-Datensatz in einen Pandas DataFrame."""
    return pd.read_csv(path)


def prepare_train_test(df: pd.DataFrame):
    """
    Trennt Features und Label und erstellt einen stratifizierten
    Train/Test-Split für die Klassifikation.
    """
    X = df[FEATURES].copy()
    y = df[LABEL].copy()

    return train_test_split(
        X, y,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify = y  # wichtig bei unbalancierten Klassen
    )


def scale_features(X_train, X_test):
    """
    Skaliert die Features mit StandardScaler.
    Fit nur auf Trainingsdaten, um Data Leakage zu vermeiden.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled



# Modell
def build_model(input_dim: int) -> keras.Model:
    """
    Erstellt ein sequentielles neuronales Netz mit
    zwei Hidden Layers à 60 Neuronen.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(60, activation="relu"),
        keras.layers.Dense(60, activation="relu"),
        keras.layers.Dense(3, activation="softmax")  # 3 Klassen
    ])
    return model



# Visualisierung
def plot_training_history(history):
    """
    Visualisiert den Trainings- und Validierungs-Loss,
    um Overfitting zu erkennen.
    """
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="test loss")
    plt.xlabel("epochs")
    plt.ylabel("loss (sparse cross entropy)")
    plt.grid(True)
    plt.legend()
    plt.show()



# Main Pipeline
def main():
    # Anzeigeoptionen für Pandas
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)

    # Daten laden
    df = load_data(CSV_PATH)
    print(df.head())

    # Klassenverteilung prüfen (wichtig bei Klassifikation)
    print("\nKlassenverteilung:")
    print(df[LABEL].value_counts().sort_index())

    # Train/Test-Daten vorbereiten
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    # Feature-Skalierung
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Modell erstellen und kompilieren
    model = build_model(input_dim=X_train_scaled.shape[1])
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Modell trainieren
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=0
    )

    # Vorhersagen auf Testdaten
    y_pred = model.predict(X_test_scaled).argmax(axis=1)

    # Bewertung des Modells
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Overfitting-Plot
    plot_training_history(history)


# Startpunkt des Skripts
if __name__ == "__main__":
    main()
