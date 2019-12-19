# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:29:57 2018

@author: anand
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier

df_fruits=pd.read_excel(r"D:\Work\DataScience\DS ppt\datascience notes\DS-5\fruits_colours.xlsx")

X=df_fruits[["mass","width","height","color_score"]]
y=df_fruits.fruit_name

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
model=KNeighborsClassifier(n_neighbors=11)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
model=KNeighborsClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
load_data=datasets.load_boston()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
model=KNeighborsRegressor()
model.fit(X_train,y_train)
print(model.predict(X_test))
print("Train Score: ",model.score(X_train,y_train))
print("Test Score: ",model.score(X_test,y_test))