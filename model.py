# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:50:12 2021

@author: SAMSUNG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


salary_data = pd.read_csv("F:\Courses\TCS iOn\website\Salarydata.csv")
salary_data.head(2)

salary_data.shape


#1
salary_data.isnull().sum()
#2
sns.boxplot(  y="age", data=salary_data,  orient='v')
plt.show()

Q1=np.percentile(salary_data["age"],25,interpolation="midpoint")
print("Q1)",Q1)
Q2=np.percentile(salary_data["age"],50)
print("Q2",Q2)
Q3=np.percentile(salary_data["age"],75)
print("Q3",Q3)
IQR=Q3-Q1
print("IQR",IQR)
low_lim=Q1-1.5*IQR
upp_lim=Q3+1.5*IQR
print("low_lim",low_lim)
print("upp_lim",upp_lim)
outlier=[]
for x in salary_data["age"]:
    if (x>upp_lim) or (x<low_lim):
        outlier.append(x)

ind_list= []
ind3= salary_data['age']>upp_lim
ind_list.append(list(salary_data.loc[ind3].index))
ind_list[0]

salary_data.drop([19172,
 19180,
 19212,
 19489,
 19495,
 19515,
 19689,
 19747,
 19828,
 20249,
 20421,
 20463,
 20482,
 20483,
 20610,
 20826,
 20880,
 20953,
 21343,
 21501,
 21812,
 21835,
 22220,
 22481,
 22895,
 22898,
 23459,
 23900,
 24027,
 24043,
 24238,
 24280,
 24395,
 24560,
 25163,
 25303,
 25397,
 26012,
 26242,
 26731,
 27795,
 28176,
 28463,
 28721,
 28948,
 29594,
 29724,
 31030,
 31432,
 31696,
 31814,
 31836,
 31855,
 32277,
 32367,
 32459,
 32494,
 32525])

salary_data.rename(columns = {'marital-status':'maritalstatus'}, inplace = True)
salary_data.rename(columns = {'native-country':'nativecountry'}, inplace = True)



from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
salary_data["workclass"]=encoder.fit_transform(salary_data["workclass"])
salary_data["education"]=encoder.fit_transform(salary_data["education"])
salary_data["maritalstatus"]=encoder.fit_transform(salary_data["maritalstatus"])
salary_data["occupation"]=encoder.fit_transform(salary_data["occupation"])
salary_data["relationship"]=encoder.fit_transform(salary_data["relationship"])
salary_data["race"]=encoder.fit_transform(salary_data["race"])
salary_data["sex"]=encoder.fit_transform(salary_data["sex"])
salary_data["nativecountry"]=encoder.fit_transform(salary_data["nativecountry"])
salary_data["salary"]=encoder.fit_transform(salary_data["salary"])


#standard scaling
x_data=salary_data.drop(["salary"],axis=1)
y_data=salary_data["salary"]
print(x_data.shape)
print(y_data.shape)
x_data_1=x_data.copy()
x_data_1.head(2)
#standard scaling
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_data=scalar.fit_transform(x_data)
x_data=pd.DataFrame(x_data,columns=x_data_1.columns)
x_data.head(3)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,random_state=42,test_size=0.2)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

#pip install catboost

from sklearn.metrics import make_scorer, accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
clf = CatBoostClassifier()


cat_b = CatBoostClassifier(iterations=500,
                           loss_function='Logloss',
                           depth=5,
                           eval_metric='Accuracy',
                           leaf_estimation_iterations=10,
                           logging_level='Silent',
                           random_seed=42
                          )


cat_b.fit(x_train,y_train)


pickle.dump(cat_b,open('model.pkl','wb') )
