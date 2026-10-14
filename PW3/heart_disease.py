import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

df = pd.read_csv("heart.csv")

print("Dataset shape:", df.shape)

print(df.head())

print(df["chol"])

print(df.describe())

columns = ["age", "chol", "trestbps", "thalach"]
data = df[columns]

print(data.describe())


fig, axes = plt.subplots(2, 2, figsize = (12, 8))

axes = axes.flatten()

for i, column in enumerate(columns):
    axes[i].hist(data[column], bins=15, edgecolor="black")
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("distributions.png")
plt.close()

# checking for normality using Q-Q plots
fig, axes = plt.subplots(2, 2, figsize = (12, 8))

axes = axes.flatten()

for i, column in enumerate(columns):
    stats.probplot(data[column].dropna(), dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q plot: {column}")

plt.tight_layout()
plt.savefig("qq_plots.png")
plt.close()

# Shapiro Wilk test
print("Shapiro-Wilk test: ")
for column in columns:
    print()
    values = data[column].dropna()

    W, p = stats.shapiro(values)

    print(f"\nColumn: {column}")
    print(f"P-value: {p:.6f}")
    if p < 0.05:
        print("Verdict: evidence against normality -> Mann-Whitney U")
    else:
        print("Verdict: insufficient evidence against normality")


#Drawing conclusions

#task 4
healthy = df[df["target"] == 0]["thalach"].dropna()
disease = df[df["target"] == 1]["thalach"].dropna() #separating healthy and unhealthy patients into 2 groups

healthy_mean = healthy.mean()
disease_mean = disease.mean() #calculating means

print("\nMean maximum heart rate:")
print("Healthy:", healthy_mean)
print("Heart disease:", disease_mean)

healthy_sem = healthy.sem()
disease_sem = disease.sem() #calculating standard errors

print("\nStandard error of the mean:")
print("Healthy:", healthy_sem)
print("Heart disease:", disease_sem)

#Matt-Whitney U test

u, p = stats.mannwhitneyu(healthy, disease)

print("\nMann-Whitney U test:")
print("P-value:", p)

if p < 0.05:
    print("Conclusion: There is a statistically significant difference.")
else:
    print("Conclusion: There is insufficient evidence of a difference.")

means = [healthy_mean, disease_mean]
errors = [healthy_sem, disease_sem]
labels = ["Healthy", "Heart disease"]

plt.bar(labels, means, yerr=errors, capsize=8)
plt.ylabel("Maximum heart rate (beats/min)")
plt.title("Maximum Heart Rate by Heart Disease Status")
plt.tight_layout()
plt.savefig("task4_heart_rate.png", dpi=300)
plt.close()

#task 5
print("\nCheking whether age and thalach rise and fall together, or in opposite directions, or not at all")
data = df[["age", "thalach"]].dropna()

age = data["age"]
thalach = data["thalach"]

r, p = stats.pearsonr(age, thalach)

print("r value: ", r)

if r > 0:
    print("Positive correlation")
elif r < 0:
    print("Negative correlation")
else:
    print("No linear correlation")

plt.scatter(age, thalach)
plt.xlabel("Age (years)")
plt.ylabel("Maximum heart rate (beats/min)")
plt.title("Age vs Maximum Heart Rate")
plt.tight_layout()
plt.savefig("task5_correlation.png", dpi=300)
plt.close()

