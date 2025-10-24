import pandas as pd
import matplotlib.pyplot as plt
import random

# Teil 3a)
subjects = ["Mathe 3", "Grundlagen der KI", "Datenanalyse und ML", "Compilerbau", "Softwaretechnik", "Wissentschaft & technisches Arbeiten"]
grades = []
for subject in subjects:
    grades.append(random.randint(1, 5))

# Teil 3b)
grades2 = []
for subject in subjects:
    grades2.append(random.randint(1, 5))


grade_series = pd.Series(grades, index=subjects, name="Notenübersicht 1. Halbjahr")
print("Series 1 Ausgabe für Aufgabe 3a)")
print(grade_series)
grade_series2 = pd.Series(grades2, index=subjects, name="Notenübersicht 2. Halbjahr")
print("\nSeries 2 Ausgabe für Aufgabe 3b)")
print(grade_series2)

average_grade = (grade_series + grade_series2) / 2

print("\nDurchschnitt aller Fächer:\n", average_grade)

