import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#section-1
df= pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head(5)
df.describe()
df.info()
Numerical_features=df.select_dtypes(include=[np.number]).columns
Categorical_features=df.select_dtypes(include="object").columns
print("Numerical_Features:",Numerical_features)
print("Categorical_Features:",Categorical_features)
df.isnull().sum()
print(df["Attrition"].value_counts())

#section-B
x=df.drop("Attrition", axis=1)
y=df["Attrition"]

y=y.map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.2,random_state=42)

#for label encoding
X_train_label=X_train.copy()
X_test_label=X_test.copy()
le=LabelEncoder()
for cols in X_train_label.select_dtypes(include="object").columns:
    le.fit(X_train_label[cols])
    X_train_label[cols]=le.transform(X_train_label[cols])
    X_test_label[cols]=le.transform(X_test_label[cols])

model_label=LogisticRegression(max_iter=8000)
model_label.fit(X_train_label,y_train)

y_pred_label=model_label.predict(X_test_label)
accuracy_label=accuracy_score(y_test, y_pred_label)
print(accuracy_label)

#onehot encoding
X_train_ohe=pd.get_dummies(X_train, drop_first=True)
X_test_ohe=pd.get_dummies(X_test, drop_first=True)

X_train_ohe, X_test_ohe = X_train_ohe.align(
    X_test_ohe, join="left", axis=1, fill_value=0
)

model_ohe=LogisticRegression(max_iter=8000)
model_ohe.fit(X_train_ohe,y_train)

y_pred_ohe=model_ohe.predict(X_test_ohe)
accuracy_ohe=accuracy_score(y_test, y_pred_ohe)
print(accuracy_ohe)

#standard scaling
scalor_std=StandardScaler()
X_train_std=scalor_std.fit_transform(X_train_ohe)
X_test_std=scalor_std.transform(X_test_ohe)

model_std=LogisticRegression(max_iter=8000)
model_std.fit(X_train_std, y_train)

y_pred_std=model_std.predict(X_test_std)
accuracy_std=accuracy_score(y_test, y_pred_std)
print("Accuracy after std_scaling: ", accuracy_std)
#MinMax scaling
scalor_MinMax=MinMaxScaler()
X_train_MinMax=scalor_MinMax.fit_transform(X_train_ohe)
X_test_MinMax=scalor_MinMax.transform(X_test_ohe)

model_MinMax=LogisticRegression(max_iter=5000)
model_MinMax.fit(X_train_MinMax, y_train)

y_pred_MinMax=model_MinMax.predict(X_test_MinMax)
accuracy_MinMax=accuracy_score(y_test, y_pred_MinMax)
print("Accuracy after MinMax_scaling",accuracy_MinMax)

comparison = pd.DataFrame({
    "Method": [
        "Label Encoding",
        "One-Hot Encoding",
        "StandardScaler",
        "MinMaxScaler"
    ],
    "Accuracy": [
        accuracy_label,
        accuracy_ohe,
        accuracy_std,
        accuracy_MinMax
    ]
})

print("\nFinal Comparison Table:\n")
print(comparison)

print("\nConclusion:")
print("One-Hot Encoding generally performs better (not in this case) than Label Encoding.")
print("Scaling improves Logistic Regression performance.")
print("Here MinMax scalor usually works better for this dataset for logistic regression.")
