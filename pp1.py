import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


df=pd.read_excel("/Users/ayushparoha/Documents/Data_Science/env/college/prac1/AirQualityUCI.xlsx")
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.loc[4:11, ["PT08.S1(CO)",'PT08.S2(NMHC)']]=np.nan
df.isnull().sum()
df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
df["Time"]=pd.to_datetime(df["Time"], errors="ignore")
cols = ["PT08.S1(CO)", "PT08.S2(NMHC)"]
df[cols] = df[cols].fillna(df[cols].mode().iloc[0])
df.isnull().sum()
col=df.columns.tolist()

num_col=df.select_dtypes(include=np.number).columns.tolist()
print(num_col)
df[num_col]=pd.to_numeric(num_col, errors="coerce")

#df.dropna(axis=0, subset=None, inplace=False)
df = df.dropna(subset=["NOx(GT)"])#not in actual file
df.replace(-200, np.nan, inplace=True)

df.columns

def analyze_column(column_name):

  print("\n=====================")
  print("Analysis of:", column_name)
  print("=======================\n")

  print("Summary Statistics:")
  print(df[column_name].describe())

  print("\nKey Measures:")
  print("Mean:", df[column_name].mean())
  print("Standard deviation:", df[column_name].std())
  print("Skewness:", df[column_name].skew())

  plt.figure()
  plt.hist(df[column_name].dropna(), bins=20)
  plt.title("Histogram of" + column_name)
  plt.xlabel(column_name)
  plt.ylabel("Frequency")
  plt.show()

  plt.figure()
  plt.boxplot(df[column_name].dropna())
  plt.title("Boxplot of" + column_name)
  plt.show()

# Convert problematic columns to numeric
for col in ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']:
    # Check if the column is of object type and contains commas before converting
    # This prevents errors if a column is already numeric or doesn't have commas
    if df[col].dtype == 'object' and df[col].astype(str).str.contains(',').any():
        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)


cols = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
for col in cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
analyze_column('CO(GT)')
analyze_column('RH')
analyze_column('T')
analyze_column('AH')