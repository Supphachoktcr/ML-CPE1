import pandas as pd
import matplotlib.pyplot as plt

# โหลดข้อมูล
columns = [
    "RI", "Na", "Mg", "Al", "Si",
    "K", "Ca", "Ba", "Fe", "Type"
]

df = pd.read_csv("glass.csv", header=None, names=columns)

# Histogram
df.hist(figsize=(12, 8))
plt.suptitle("Histogram of Glass Dataset")
plt.show()

import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()