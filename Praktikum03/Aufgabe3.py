import pandas as pd

# Aufgabenteil a)
weekdays = pd.date_range(start='2025-12-24', end='2026-01-06', freq='B')    # freq B = Business Day
print("Aufgabenteil a - reguläre Wochentage")
print(weekdays)
print("Es gibt", len(weekdays), "normale Wochentage/reguläre Arbeitstage in den Weihnachtsferien 2025/26.")

# Aufgabenteil b)
sundays = pd.date_range(start='2025-12-24', end='2027-01-06', freq='W')     # freg W = Ein bestimmter Tag, Standard Sonntag
sundays_on_first = sundays[sundays.day == 1]
print("\nAufgabenteil b - Sonntage am 01. des Monats")
print(sundays_on_first)
print("Vom 24.12.2025 bis zum 6.01.2027 gibt es", len(sundays_on_first), "Sonntage, die auf den 1. des Monats fallen.")