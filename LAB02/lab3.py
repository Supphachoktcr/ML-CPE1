import pandas as pd

# โหลดข้อมูล
columns = [
    "RI", "Na", "Mg", "Al", "Si",
    "K", "Ca", "Ba", "Fe", "Type"
]

df = pd.read_csv("glass.csv", header=None, names=columns)

# 1. Missing Value Handling
print("===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Duplicate Records Before =====")
print(df.duplicated().sum())

# ลบข้อมูลซ้ำ
df = df.drop_duplicates()

print("\n===== Duplicate Records After =====")
print(df.duplicated().sum())

print("\n===== Incorrect Data Check =====")
print(df[df["Na"] < 0])

print("\n===== Data Types Before =====")
print(df.dtypes)

df["Type"] = df["Type"].astype(int)

print("\n===== Data Types After =====")
print(df.dtypes)

print("\n===== Mean =====")
print(df.mean(numeric_only=True))

print("\n===== Median =====")
print(df.median(numeric_only=True))