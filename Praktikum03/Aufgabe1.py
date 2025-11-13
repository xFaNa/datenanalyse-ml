import pandas as pd

# CSV einlesen
df = pd.read_csv(
    "countries_population.csv",
    sep= r"\s+",            # Trennung nach Leerzeichen
    thousands= ",",         # Entfernt Tausenderkommas
    quotechar= "'",         # Liest 'China' als China
    header= None,           # Es gibt keine Kopfzeile
    names= ["Land", "Einwohner"] # Eigene Kopfzeile
)

df = df.set_index("Land")

print(df.head())