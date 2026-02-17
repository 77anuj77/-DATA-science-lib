import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer

d= "heart.csv"
data=pd.read_csv(d)
print(data.head())

df=pd.DataFrame(data)
print(df.head())

print(df.shape)

d=df["chol"].isnull
print(d)

print(df["chol"].nunique)

print(df.describe())
print(df.info())

df_missing=df

df_missing=df.copy()
df_missing.loc[5:15, "chol"]=np.nan
df_missing.loc[4:14, "trestbps"]=np.nan
print(df_missing[["chol","trestbps"]].isnull().sum())

df_missing["trestbps"].fillna(df_missing["trestbps"].median, inplace=True)
print(df_missing.head())
df_missing['chol'].fillna(df_missing['chol'].mode, inplace=True)
print(df_missing.head())

print(df_missing.isnull().sum())
df_missing.loc[3:3,"sex"]=np.nan
print(df_missing)

df_missing["sex"].fillna(df_missing["sex"].mode,inplace=True)
print(df_missing)


df_knn= df_missing.copy()
df_knn.loc[5:15 ,["chol"]]=np.nan
df_knn.loc[0:8, ["trestbps"]]=np.nan
print(df_knn.head())

knn_imputer=KNNImputer(n_neighbors=5)
df_knn=pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
print(df_knn.head())

print(df_knn.isnull().sum())