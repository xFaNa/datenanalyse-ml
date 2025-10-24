import pandas as pd

# Aufgabenteil a)
cities = {
    "Wien": {
        "country": "Österreich",
        "area": 414.6,
        "population": 1805681
    },
    "Berlin": {
        "country": "Deutschland",
        "area": 891.85,
        "population": 3562166
    },
    "Zürich": {
        "country": "Schweiz",
        "area": 87.88,
        "population": 378884
    }
}

df = pd.DataFrame(cities)

for city in df.columns:
    print(f"{city:<10} country      {df.loc['country', city]}")
    print(f"{'':<10} area         {df.loc['area', city]}")
    print(f"{'':<10} population   {df.loc['population', city]}")

