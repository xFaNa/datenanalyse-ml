import pandas as pd

df = pd.DataFrame(
    {
        "country": ["Österreich", "Deutschland", "Schweiz"],
        "area": [414.6, 891.85, 87.88],
        "population": [1805681, 3562166, 378884],
    },
    index=["Wien", "Berlin", "Zürich"]
)

s = df.stack()
print(s)

# Aufgabenteil b)
print("\nAlphabetisch sortiert.")
s_sorted = s.sort_index()
print(s_sorted)

# Aufgabenteil c)
print("\nVertauschte Indizes")
s_swapped = s.swaplevel()
s_swapped = s_swapped.sort_index()
print(s_swapped)