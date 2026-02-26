#categorical data- we have already learned 
#numerical data-
#encoding-:> we will covert the categorical data to numerical data
#label encodiig->i will assign numbers to each and different data,
                # Eg.-roe of red-1, yellow-2, blue-3
                # it will automatically give priority to colors( 3>2>1 )
                # Fast and Efficiet and less memory usage
#on_hot encoding->for every red green blue i will devide it into three different cols
                # and give 1 to red and other 0 at a time 
                # slow but accurate
#churn- the employees is not with the comany or with the company

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


df=pd.read_csv("Telco_Customer.csv")
print(df.head())

print(df.isnull().sum())
print(df.info())

x=df.drop("Churn", axis=1)
y=df["Churn"]

df_label=df.copy()
le=LabelEncoder()

#fit_tranform()->> converts the categorical data to numerical data 
for col in df_label.select_dtypes(include='object').columns:
    df_label[col]=le.fit_transform(df_label[col])

df_label.head()

x_label=df_label.drop('Churn', axis=1)
y_label=df_label['Churn']

#random_state=42 here 42 is just a number whereas random_state is used for using the same slot of testing many times
x_train, x_test, y_train, y_test= train_test_split(x_label,y_label,test_size=0.2, random_state=42)

#training
#max_iter=8000: model will run 8000 times and if 7000 is yes 1000 is no then answer is yes
# the values after each run of model will be saved using fit
model_label=LogisticRegression(max_iter=8000)
model_label.fit(x_train, y_train)

#testing 
y_pred_label= model_label.predict(x_test)
accuracy_label=accuracy_score(y_test, y_pred_label)

print(accuracy_label)

#get_dummies is used for getting 1 and 0 which are there in onehot
'''drop_first= True green blue red
                    1      0    0
                    0      0    1
                    -      -    -
                    then automatically it will put 1 in blue column
                    what matters is the colours '''

df_onehot=pd.get_dummies(df, drop_first=True)
df_onehot.head()

#training
x_onehot=df_onehot.drop('Churn_Yes', axis=1)
y_onehot=df_onehot['Churn_Yes']

x_train, x_test, y_train, y_test= train_test_split(x_onehot,y_onehot,test_size=0.2, random_state=42)

model_onehot=LogisticRegression(max_iter=8000)
model_onehot.fit(x_train, y_train)

y_pred_onehot=model_onehot.predict(x_test)
accuracy_onehot=accuracy_score(y_test, y_pred_onehot)

print(accuracy_onehot)

#to show these accuracy of both we make comparision table
comparision= pd.DataFrame({
    'Encoding Technique':['Label Encoding', 'One-hot Encoding'],
    'Accuracy':[accuracy_label, accuracy_onehot]
})
print(comparision)


#feature scalling ->> to solve the mixed data(categorical and numerical) issue we use normalisation
#standard scaling->> standardise data by making mean=0 & SD=1 [xnew= x-mean/sd]
#min-max scaling ->>scales the data into a range between 0,1 [xnew=x-xmin/xmax-xmin]

scaler_std=StandardScaler()
x_train_std= scaler_std.fit_transform(x_train)
x_test_std=scaler_std.transform(x_test)

model_std= LogisticRegression(max_iter=1000)
model_std.fit(x_train_std, y_train)

y_pred_std=model_std.predict(x_test_std)
accuracy_std=accuracy_score(y_test, y_pred_std)

print(accuracy_std)


scaler_mm=MinMaxScaler()
x_train_mm=scaler_mm.fit_transform(x_train)
x_test_mm= scaler_mm.transform(x_test)

model_mm= LogisticRegression(max_iter=1000)
model_mm.fit(x_train_mm, y_train)

y_pred_mm=model_mm.predict(x_test_mm)
accuracy_mm= accuracy_score(y_test, y_pred_mm)

print(accuracy_mm)

#Final comparision after scalling on onehot
comparision_scalling=pd.DataFrame({
    'Method':['Original(one-hot)', 'StandardScaler', 'MinMaxScaler'],
    'Accuracy':[accuracy_onehot, accuracy_std, accuracy_mm]
})
print(comparision_scalling)

#standard scaler is used in ML-Models training but not necessary
#standard scaler is much better as compared to minmax because for bigger
#data it is not possible for every value to lie in 0,1