import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris # A convenient way to get standard datasets

# 1. Data Loading (Using Scikit-learn for convenience)
print("--- 1. Loading and Initial Inspection ---")
iris_data = load_iris(as_frame=True)
df = iris_data.frame
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())
print("-" * 40)

# 2. Data Cleaning and Preparation (Simulated Missing Data)
# A crucial step: checking for missing values (nulls) and data types
print("--- 2. Data Cleaning and Statistics ---")
missing_count = df.isnull().sum().sum()
print(f"Total missing values detected (before simulation): {missing_count}")

# Simulate adding a missing value for a realistic scenario
df.loc[5, 'sepal length (cm)'] = np.nan 
print(f"Total missing values detected (after simulation): {df.isnull().sum().sum()}")

# Impute missing value using the mean of the column (a common technique)
df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)
print("Missing value imputed using column mean.")

# Group the data and calculate statistics for different species (groups)
grouped_stats = df.groupby('target_names')['petal length (cm)'].agg(['mean', 'std', 'count'])
print("\nPetal Length Statistics Grouped by Species:")
print(grouped_stats)
print("-" * 40)

# 3. Data Visualization (Matplotlib)
print("--- 3. Data Visualization ---")

# Scatter Plot: visualize the relationship between two features
plt.figure(figsize=(10, 5))
scatter = plt.scatter(
    df['sepal length (cm)'], 
    df['petal length (cm)'], 
    c=df['target'], # Color-code by species (target)
    cmap='viridis'
)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Sepal Length vs. Petal Length (Colored by Species)')
plt.legend(*scatter.legend_elements(), title="Species")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Histogram: visualize the distribution of a single feature
plt.figure(figsize=(7, 5))
plt.hist(df['petal width (cm)'], bins=15, edgecolor='black', alpha=0.7)
plt.title('Distribution of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.show()
