import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

#task 6

df = pd.read_csv("chemicals_cancer.csv")

print(df.shape)
print(df.head())
print(df.columns)
print(df.describe())
print()

data = df[["benzene", "cadmium", "malignancy"]].dropna()
 
benzene = data["benzene"]
cadmium = data["cadmium"]
malignancy = data["malignancy"]
 
r_benzene, p = stats.pearsonr(benzene, malignancy)
r_cadmium, p = stats.pearsonr(cadmium, malignancy)
 
print("benzene vs malignancy, r value: ", r_benzene)
print("cadmium vs malignancy, r value: ", r_cadmium)
 
if abs(r_benzene) > abs(r_cadmium):
    print("At face value, benzene looks like the cause")
else:
    print("At face value, cadmium looks like the cause")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(benzene, malignancy, s=8)
axes[0].set_xlabel("Benzene")
axes[0].set_ylabel("Malignancy score")
axes[0].set_title("Benzene vs Malignancy")

axes[1].scatter(cadmium, malignancy, s=8)
axes[1].set_xlabel("Cadmium")
axes[1].set_ylabel("Malignancy score")
axes[1].set_title("Cadmium vs Malignancy")

plt.tight_layout()
plt.savefig("task6_scatterplots.png", dpi=300)
plt.close()

#task 7

print()
print(df[["benzene", "cadmium", "malignancy", "pollution_index", "age"]].corr())


similar = df[(df["pollution_index"] > 40) & (df["pollution_index"] < 60)]
similar = similar.dropna()
print("\nPatients with pollution index between 40 and 60: ", len(similar))
 
r_benzene2, p = stats.pearsonr(similar["benzene"], similar["malignancy"])
r_cadmium2, p = stats.pearsonr(similar["cadmium"], similar["malignancy"])

print("\nbenzene: r before =", r_benzene, " r after =", r_benzene2)
print("cadmium: r before =", r_cadmium, " r after =", r_cadmium2)

if abs(r_benzene2) < abs(r_benzene) / 2:
    print("\nBenzene's association mostly disappeared -> it only looked guilty")
else:
    print("\nBenzene's association survived")

if abs(r_cadmium2) < abs(r_cadmium) / 2:
    print("Cadmium's association mostly disappeared -> it only looked guilty")
else:
    print("Cadmium's association survived")

#bonus task

print("\nBonus: entropy of the target column")
 
heart = pd.read_csv("heart.csv")
p = heart["target"].value_counts(normalize=True)
print(p)
 
entropy = -(p * np.log2(p)).sum()
print("Entropy (bits): ", entropy)
print("Maximum possible for 2 categories: 1 bit")