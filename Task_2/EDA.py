import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

# Change path to your CSV file
df = pd.read_csv(r"F:\python\Task 2\titanic.csv")   # example: Titanic dataset

# 2. Basic Information

print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)

# 3. Missing Values

print("\nMissing Values:\n", df.isnull().sum())

# 4. Statistical Summary

print("\nStatistical Summary:\n", df.describe())

# 5. Duplicate Rows

print("\nDuplicate Rows:", df.duplicated().sum())

# 6. Correlation Heatmap

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 7. Distribution of Numerical Columns

df.hist(figsize=(12, 7))
plt.suptitle("Numerical Feature Distributions")
plt.show()

# 8. Automated EDA Report

profile = ProfileReport(
    df,
    title="EDA Report",
    explorative=True
)

profile.to_file("EDA_Report.html")

print("\nEDA Report Generated Successfully: EDA_Report.html")
