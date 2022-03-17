# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 15:02:22 2021

@author: SAMSUNG
"""

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


salary_data.isnull().sum()
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
                           depth=6,
                           eval_metric='Accuracy',
                           leaf_estimation_iterations=10,
                           logging_level='Silent',
                           random_seed=42
                          )


cat_b.fit(x_train,y_train)

y_pred=cat_b.predict(x_test)

from sklearn.metrics import f1_score,confusion_matrix

f1=f1_score(y_test,y_pred)

cm=confusion_matrix(y_test,y_pred)

pd.Series(rf.feature_importances_,index=x.columns).sort_values(ascending=False)*100




from sklearn.model_selection import train_test_split    

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)    
x_train.shape
x_test.shape    
y_train.shape
y_test.shape



'''
Model builting: cat

'''
from sklearn.metrics import make_scorer, accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
clf = CatBoostClassifier()


cat_b = CatBoostClassifier(iterations=500,
                           loss_function='Logloss',
                           depth=6,
                           eval_metric='Accuracy',
                           leaf_estimation_iterations=10,
                           logging_level='Silent',
                           random_seed=42
                          )


cat_b.fit(x_train,y_train)

y_pred=cat_b.predict(x_test)

'''
checking the model performance

'''
from sklearn.metrics import f1_score,confusion_matrix

red_f1=f1_score(y_test,y_pred)

red_cm=confusion_matrix(y_test,y_pred)

pd.Series(cat_b.feature_importances_,index=x.columns).sort_values(ascending=False)*100



