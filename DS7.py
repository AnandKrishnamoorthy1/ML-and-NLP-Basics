# -*- coding: utf-8 -*-
"""
Created on Sat May  5 10:11:59 2018

@author: anand
"""

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import LinearSVC
import numpy as np

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

model=LinearSVC()
model.fit(X_train,y_train)
print(model.score(X_train,y_train).round(2))

print("Model cross val Score: ",cross_val_score(model,X_train,y_train).round(2))
print("Model cross val Mean Score: ",np.mean(cross_val_score(model,X_train,y_train)).round(2))

############################################################
#Simple decision tree
from sklearn import datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
clf=DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
predict_val=clf.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})
#print(price_compare.round(1))

print('Model Train Score: ',clf.score(X_train,y_train).round(2))
print('Model Test Score: ',clf.score(X_test,y_test).round(2))

############################################################
#Decision tree with pre-pruning and feature importance
#max_depth, max_leaf_nodes, min_leaf_samples
from sklearn import datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
clf=DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)
predict_val=clf.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})

print('Model Train Score: ',clf.score(X_train,y_train).round(2))
print('Model Test Score: ',clf.score(X_test,y_test).round(2))

############################################################

#Fearure importance
print('Model feature importances: ',clf.feature_importances_)

feature_imp=pd.DataFrame(load_data.feature_names,columns=["Features"])
feature_imp["Importances"]=clf.feature_importances_
feature_imp.sort_values(["Importances"])

###############################################################################
#Random Forest
from sklearn import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
clf=RandomForestClassifier(n_estimators=8,random_state=0,max_depth=3)
clf = clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
predict_val=clf.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})

print('Model Train Score: ',clf.score(X_train,y_train).round(2))
print('Model Test Score: ',clf.score(X_test,y_test).round(2))
############################################################

#Dummy Classifiers
from sklearn import datasets
import pandas as pd
from sklearn.dummy import DummyClassifier

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
dummy=DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
dummy_predict=dummy.predict(X_test)
print(dummy.score(X_test,y_test))
############################################################

#Random Forest->Confusion Matrix
from sklearn import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
clf=RandomForestClassifier(n_estimators=8,random_state=0,max_depth=3,n_jobs=-1)
clf = clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
predict_val=clf.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})

print('Model Train Score: ',clf.score(X_train,y_train).round(2))
print('Model Test Score: ',clf.score(X_test,y_test).round(2))
print(confusion_matrix(y_test, predict_val))
###############################################################################

#Random Forest->Precision and recall
from sklearn import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
clf=RandomForestClassifier(n_estimators=8,random_state=0,max_depth=3,n_jobs=-1)
clf = clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
predict_val=clf.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})

print('Model Train Score: ',clf.score(X_train,y_train).round(2))
print('Model Test Score: ',clf.score(X_test,y_test).round(2))
print(confusion_matrix(y_test, predict_val))
print(precision_score(y_test, predict_val))
print(recall_score(y_test, predict_val))
###############################################################################

#Random Forest->ROC
from sklearn import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
clf=RandomForestClassifier(n_estimators=8,random_state=0,max_depth=3,n_jobs=-1)
clf = clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
predict_val=clf.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})

print('Model Train Score: ',clf.score(X_train,y_train).round(2))
print('Model Test Score: ',clf.score(X_test,y_test).round(2))
prob_score=pd.DataFrame(clf.predict_proba(X_test),columns=["class1","class2"])
print(roc_auc_score(y_test, prob_score.class2))
###############################################################################

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

load_data=datasets.load_boston()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

model=LinearRegression().fit(X_train,y_train)
dummy=DummyRegressor(strategy='mean').fit(X_train,y_train)

model_predict=model.predict(X_test)
dummy_predict=dummy.predict(X_test)

print(r2_score(y_test,model_predict))
print(r2_score(y_test,dummy_predict))
###############################################################################

# Load the library with the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix

iris = load_iris()
# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=iris.target
df['species']=df['target'].replace({0:'setosa',1:'versicolor',2:'virginica'})


#Splitting to train and test
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], random_state=1)

def Neural_classifier():
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(16, 4),max_iter=2000, random_state=5)
    clf.fit(X_train,y_train)
    class_predict=clf.predict(X_test)
    print("Confusion Matrix [Neural Networks]:")
    print(pd.DataFrame(
        confusion_matrix(y_test, class_predict),
        columns=['setosa', 'versicolor','virginica'],
        index=['setosa', 'versicolor','virginica']
    ))
    
Neural_classifier() 
###############################################################################