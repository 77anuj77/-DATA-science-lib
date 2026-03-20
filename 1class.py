import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
from scipy.stats import norm

# Load dataset
df = pd.read_csv("global air pollution dataset (1).csv")

# Histogram of AQI
plt.figure()
plt.hist(df['AQI Value'], bins=20)
plt.title("Distribution of AQI Value")
plt.xlabel("AQI Value")
plt.ylabel("Frequency")
plt.show()
sns.histplot(df["AQI Value"], bins=20 ,kde=True )
plt.show

# Generate normal distributed data
data = np.random.normal(loc=0, scale=1, size=1000)

# Mean and std
mean = statistics.mean(data)
sd = statistics.stdev(data)

# X values for smooth curve
x = np.linspace(min(data), max(data), 100)

# Plot histogram
plt.hist(data, bins=20, density=True, alpha=0.6)

# Plot normal distribution curve
#scipy.ststs import norm 
#pdf=probability distribution function
plt.plot(x, norm.pdf(x, mean, sd), linewidth=222)

plt.title("Normal Distribution with PDF Curve")
plt.show()


sns.boxplot(df["NO2 AQI Value"])
plt.show()

#showfliers for detecting the outliers
#autopct=auto percentage === %1.1f%% menans ony one dcimal after point
plt.figure()
plt.boxplot(df["NO2 AQI Value"], showfliers=False)
plt.title("Boxplot of NO2 Value")
plt.ylabel("AQI Value")
plt.show()

plt.figure()
df["AQI Category"].value_counts().plot(kind="bar")
plt.title("AQI Category Distribution")
plt.xlabel("AQI Category")
plt.ylabel("Count")
plt.show()

plt.figure()
df["NO2 AQI Value"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("NO2 AQI Category Percentage")
plt.ylabel("")
plt.show()

plt.figure()
df["AQI Category"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("AQI Category Percentage")
plt.ylabel("")
plt.show()