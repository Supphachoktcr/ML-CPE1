import pandas as pd

# โหลดไฟล์ CSV
columns = [
    "RI", "Na", "Mg", "Al", "Si",
    "K", "Ca", "Ba", "Fe", "Type"
]

df = pd.read_csv("glass.csv", header=None, names=columns)

print(df)

print("\nShape:")
print(df.shape)

print("\n===== Shape =====")
print(df.shape)

print("\n===== Data Types =====")
print(df.dtypes)

print("\n===== Summary Statistics =====")
print(df.describe())

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Duplicate Records =====")
print(df.duplicated().sum())

print("\n===== Class Distribution =====")
print(df["Type"].value_counts())
