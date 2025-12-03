import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set(style="whitegrid")
pd.set_option("display.max_columns", None)          # alle Spalten anzeigen
pd.set_option("display.expand_frame_repr", False)   # Zeilen nicht umbrechen

# Daten laden

df = pd.read_csv("rawdata_luftqualitaet.csv")

# a) Visualisieren + Kennzahlen
print("Erste 10 Zeilen der Datentabelle")
print(df.head(10))

# numerische Spalten (inkl. state_air_quality)
numeric_cols = df.select_dtypes(include=[np.number]).columns

stats = df[numeric_cols].agg(['min', 'max', 'mean', 'std', 'count']).T
stats = stats.rename(columns={
    'min': 'Minimum',
    'max': 'Maximum',
    'mean': 'Mittelwert',
    'std': 'Standardabweichung',
    'count': 'Anzahl_Messwerte'
})

print("\nStatistische Kennzahlen pro numerischer Spalte")
print(stats)

# Liniendiagramm (Ausschnitt)
subset = df[numeric_cols].iloc[:500]   # erste 500 Messungen

plt.figure(figsize=(12, 8))
for col in numeric_cols:
    if col != "state_air_quality":     # Zielvariable nicht als kontinuierliche Linie
        plt.plot(subset.index, subset[col], label=col, alpha=0.8)

plt.title("Zeitlicher Verlauf der Messwerte (Ausschnitt, erste 500 Messungen)")
plt.xlabel("Messindex")
plt.ylabel("Messwert")
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap der Korrelationen
corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Korrelations-Heatmap der numerischen Variablen")
plt.tight_layout()
plt.show()

# Scattermatrix
axs = scatter_matrix(df[numeric_cols],  figsize=(12, 12), diagonal='hist')
plt.suptitle("Scattermatrix der numerischen Variablen")
plt.tight_layout()
plt.show()


# b) Train-/Test-Split 80% / 20%
# Features (Eingangswerte, nur Messdaten) und Zielvariable trennen
X = df.drop("state_air_quality", axis=1)
y = df["state_air_quality"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% Test
    random_state=42,
    stratify=y          # Klassenverteilung in Train/Test ähnlich halten
)

print("\nTrain-/Test-Split")
print(f"Anzahl Samples gesamt: {len(df)}")
print(f"Trainingsdaten: {X_train_raw.shape[0]} Samples")
print(f"Testdaten:      {X_test_raw.shape[0]} Samples")


# c) Normalisieren auf [0, 1] + Visualisierung zur Kontrolle
scaler = MinMaxScaler()   # bringt jede Spalte auf [0, 1]

# auf Trainingsdaten fitten, dann Testdaten transformieren
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Für Kontroll-Visualisierung wieder DataFrame bauen
X_train_norm_df = pd.DataFrame(X_train, columns=X.columns)

print("\nErste 10 Zeilen der normalisierten Trainingsdaten")
print(X_train_norm_df.head(10))

print("\nStatistik der normalisierten Trainingsdaten")
print(X_train_norm_df.describe())   # min≈0, max≈1 pro Spalte

# Kontroll-Plot: Liniendiagramm der normalisierten Werte (Ausschnitt)
subset_norm = X_train_norm_df.iloc[:500]

plt.figure(figsize=(12, 8))
for col in X_train_norm_df.columns:
    plt.plot(subset_norm.index, subset_norm[col], label=col, alpha=0.8)

plt.title("Normalisierte Messdaten (Trainingsdaten, Ausschnitt, [0, 1])")
plt.xlabel("Index im Trainingsdatensatz")
plt.ylabel("Normalisierter Wert")
plt.legend()
plt.tight_layout()
plt.show()

# Boxplot der normalisierten Features
plt.figure(figsize=(8, 6))
sns.boxplot(data=X_train_norm_df)
plt.title("Boxplot der normalisierten Trainingsdaten (Features in [0, 1])")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# d) MLP-Klassifikator trainieren und Prognosen ausgeben
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # Architektur für den kleinen Datensatz
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

# Prognose auf dem Testdatensatz
y_pred = mlp.predict(X_test)

print("\nBeispielhafte Prognosen (erste 20 Test-Samples)")
prognose_df = pd.DataFrame({
    "y_test_original": y_test.iloc[:20].values,
    "y_pred": y_pred[:20]
})
print(prognose_df)

# e) Evaluation auf dem Testdatensatz (Accuracy etc.)
acc = accuracy_score(y_test, y_pred)
print(f"\nGenauigkeit (Accuracy) auf dem Testdatensatz: {acc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Confusion-Matrix zur besseren Einsicht
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Wahre Klasse")
plt.title("Confusion-Matrix MLP – Luftqualitätsklassifikation")
plt.tight_layout()
plt.show()
