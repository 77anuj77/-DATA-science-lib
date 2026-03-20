
# ASSIGNMENT 7 - HEART DATASET

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
from scipy.stats import norm

# -------------------------------
# Section A – Data Exploration


# 1. Load dataset
df = pd.read_csv("heart.csv")
# 2. Display first 5 rows
print("\nFirst 5 rows:\n", df.head())

# 3. Info and Describe
print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# 4. Identify categorical & numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=[np.number]).columns

print("\nCategorical Columns:", list(categorical_cols))
print("Numerical Columns:", list(numerical_cols))

# 5. Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# Section B – Data Cleaning


# 1 & 2. Handle missing values
df = df.dropna() 

# 3. Confirm dataset is clean
print("\nAfter Cleaning (Missing Values):\n", df.isnull().sum())

# -------------------------------
# Section C – Statistical Analysis


# Selected columns 
cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

print("\n--- Statistical Analysis ---")

for col in cols:
    print(f"\nColumn: {col}")
    
    # Mean
    mean_val = df[col].mean()
    print("Mean:", mean_val)
    
    # Standard Deviation
    std_val = df[col].std()
    print("Standard Deviation:", std_val)
    
    # Skewness
    skew_val = df[col].skew()
    print("Skewness:", skew_val)

# -------------------------------
# Section D – Histogram


for col in cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    
    # Normal distribution curve
    mean = df[col].mean()
    std = df[col].std()
    x = np.linspace(df[col].min(), df[col].max(), 100)
    plt.plot(x, norm.pdf(x, mean, std))
    
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# -------------------------------
# Section E – Boxplot

for col in cols:
    plt.figure()
    sns.boxplot(x=df[col])
    
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

# -------------------------------
# Section F – Interpretation


means = df[cols].mean()
stds = df[cols].std()
skews = df[cols].skew()

print("\n--- Interpretation ---")

print("\nHighest Mean:", means.idxmax(), "=", means.max())
print("Highest Std Dev:", stds.idxmax(), "=", stds.max())
print("Most Skewed:", skews.idxmax(), "=", skews.max())

# Outliers check using IQR
outliers_count = {}

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outliers_count[col] = len(outliers)

max_outlier_col = max(outliers_count, key=outliers_count.get)

print("Most Outliers:", max_outlier_col, "=", outliers_count[max_outlier_col])

print("\nConclusion:")
print("The dataset shows variation in health parameters. Some variables are skewed and contain outliers, indicating non-uniform distribution. Proper preprocessing is important before modeling.")