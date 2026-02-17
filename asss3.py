import pandas as pd
import numpy as np

url="/Users/ayushparoha/Documents/Data_Science/env/college/ass3/synthetic_dataset.csv"
df=pd.read_csv(url)

print(df.head())

print(df.shape)
print(df.info())

print(df.duplicated().sum())
df.drop_duplicates()
df.shape

df.dtypes
df.isnull().sum()
df["Rating"]=df["Rating"].astype("float64")

df["Category"]=df["Category"].astype("object")
df.loc[:, ["Category", "Stock","Rating"]] = (df.loc[:, ["Category", "Stock","Rating"]].fillna("Not available"))
df[["Price"],["Rating"],["Stock"],["Discount"]]= df[["Price"],["Rating"],["Stock"],["Discount"]].
print(df["Price"].head())



df["Discount"]=df["Discount"].fillna(df["Discount"].mean(), inplace=True)

df.loc[:, "Discount"] = df["Discount"].fillna(df["Discount"].mean())
df.loc[:, "Price"] = df["Price"].fillna(df["Price"].mean())

df.duplicated()
df.isnull().sum()
df.dtypes
print(df)