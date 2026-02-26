#🟢Wrangling- to convert raw data into more meaningfull data by using stacks to organise
#🟢column/row opperation
#🟢grouping, sorting, ranking
#🟢merging multiple student datasets
'''
🔴descitization- takes raw continous data and convert(breakdown) it into diffrent categories 
🔴binning- after breakdown put it into a seperate bins 
    🟡Width-wise:-->i can give the fix width
    🟡Frequency-wise:-->distribute the data in the bins with equal values(frequency)
    (]
🔴categorisation- it names the bins made during the binning
''' 
#ranking- is to rank priorities ( like if i give 1,3 ,4 then it will only assign rank {1,3,4})
#---> to solve this problem we use dense(it awill assign the next rank to be 1,2,3 instead of 1,3,4)
#contatination->>> it dosent ask for the primary key
#merging->>> it ask for the primary key

import pandas as pd
import numpy as np

df=pd.read_csv("/Users/ayushparoha/Documents/Data_Science/env/college/ass5/StudentsPerformance.csv")
print(df.head())
print(df.describe())
print(df.info())

#renaming columns for clarity
df=df.rename(columns={
    'gender':'Gender',
    'race/ethnicity':'Ethnicity',
    'parental level of education':'Parental_educarion',
    'lunch':'Lunch_type',
    'test preparation course':'Test-Prep',
    'math score':'Math_Score',
    'reading score':'Reading_Score',
    'writing score':'Writing_score'
})
print(df.columns)
cols=df.columnst.tolist()

high_math=df[df['Math_Score']>70]
print(high_math)

#students with completed test
prep_completed=df[df["Test-Prep"]=='completed']
print(prep_completed)

df["Average_Score"]= df[["Math_Score", "Reading_Score", "Writing_score" ]].mean(axis=1)
print(df.columns)

#peformance category
df['Performance_level']=pd.cut(df['Average_Score'],bins=[0,50,75,100],labels=['low','medium','high'])
# give right=False for including 50 in medium 
print(df['Performance_level'])

print(df.columns.tolist())

avg_by_gender = df.groupby("Gender")[["Math_Score", "Reading_Score", "Writing_score"]].mean()
print(avg_by_gender)

#avg score by lunch type
avg_by_lunch=df.groupby('Lunch_type')['Average_Score'].mean()
print(avg_by_lunch)

#sorting students by avg score
sorted_df=df.sort_values(by='Average_Score', ascending=False)
print(sorted_df)

#Ranking students based on avg score
sorted_df=sorted_df['Average_Score'].rank(method='dense', ascending=False)
print(sorted_df)

#Creating the 2 dataset containing attendace information 
attendence_df=df[['Gender']].copy()
attendence_df['Attendence_Percentage']=np.random.randint(60,100, size=len(attendence_df))
print(attendence_df)

merged = pd.concat([df, attendence_df["Attendence_Percentage"]], axis=1)

high_attendance = merged[merged["Attendence_Percentage"] > 85]

print(high_attendance)

#final stuctured dataset
final_df=merged[["Gender", "Lunch_Type", "Test_Prep", "Math_Score", "Reading_Score", "Writing_score", "Average_Score", "Performance_level", "Attendence_Percentage" ]]
