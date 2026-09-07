import pandas as pd
from sklearn.preprocessing import LabelEncoder

columns = [
    "RI", "Na", "Mg", "Al", "Si",
    "K", "Ca", "Ba", "Fe", "Type"
]

df = pd.read_csv("glass.csv", header=None, names=columns)

le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

print("===== Label Encoding =====")
print(df.head())

df_onehot = pd.get_dummies(df, columns=["Type"])

print("\n===== One-Hot Encoding =====")
print(df_onehot.head())