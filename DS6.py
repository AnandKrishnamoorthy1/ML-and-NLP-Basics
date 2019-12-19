# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 10:26:06 2018

@author: anand
"""

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

load_data=datasets.load_boston()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

model=LinearRegression()
model.fit(X_train,y_train)

model_coeff=pd.DataFrame(data={'Features':load_data.feature_names,'Coefficients':model.coef_})
"""
print('Coefficients:')
print(model_coeff)
print('Intercept:')
print(model.intercept_)
"""
predict_val=model.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})
print(price_compare.round(1))

predict_train_val=model.predict(X_train)
price_compare=pd.DataFrame(data={'Actual Value':y_train,'Predicted Value':predict_train_val})
print(price_compare.round(1))

print('Model Train Score: ',model.score(X_train,y_train).round(2))
print('Model Test Score: ',model.score(X_test,y_test).round(2))


#####################################################################


from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

load_data=datasets.load_boston()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=Ridge(alpha=2)
model.fit(X_train_scaled,y_train)

model_coeff=pd.DataFrame(data={'Features':load_data.feature_names,'Coefficients':model.coef_})

print('Coefficients:')
print(model_coeff)
print('Intercept:')
print(model.intercept_)

predict_val=model.predict(X_test_scaled)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})
print(price_compare.round(1))

print('Model Train Score: ',model.score(X_train_scaled,y_train).round(2))
print('Model Test Score: ',model.score(X_test_scaled,y_test).round(2))

#####################################################################


from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

load_data=datasets.load_boston()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=Lasso(alpha=.2)
model.fit(X_train_scaled,y_train)

model_coeff=pd.DataFrame(data={'Features':load_data.feature_names,'Coefficients':model.coef_})

print('Coefficients:')
print(model_coeff)
print('Intercept:')
print(model.intercept_)

predict_val=model.predict(X_test_scaled)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})
print(price_compare.round(1))

print('Model Train Score: ',model.score(X_train_scaled,y_train).round(2))
print('Model Test Score: ',model.score(X_test_scaled,y_test).round(2))



from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

load_data=datasets.load_boston()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

poly=PolynomialFeatures()
X_train_scaled=poly.fit_transform(X_train)
X_test_scaled=poly.fit_transform(X_test)

model=Ridge(alpha=5)
model.fit(X_train_scaled,y_train)

#model_coeff=pd.DataFrame(data={'Features':load_data.feature_names,'Coefficients':model.coef_})
"""
print('Coefficients:')
print(model_coeff)
print('Intercept:')
print(model.intercept_)
"""
predict_val=model.predict(X_test_scaled)

print('Model Train Score Polynomial: ',model.score(X_train_scaled,y_train).round(2))
print('Model Test Score Polynomial: ',model.score(X_test_scaled,y_test).round(2))

#####################################################################

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

model=LogisticRegression(C=2)
model.fit(X_train,y_train)


predict_val=model.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})
print(price_compare.round(1))

print('Model Train Score: ',model.score(X_train,y_train).round(2))
print('Model Test Score: ',model.score(X_test,y_test).round(2))

#####################################################################

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

load_data=datasets.load_breast_cancer()
X=pd.DataFrame(data=load_data.data,columns=load_data.feature_names)
Y=load_data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

model=LinearSVC(C=5)
model.fit(X_train,y_train)

predict_val=model.predict(X_test)
price_compare=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_val})
print(price_compare.round(1))

print('Model Train Score: ',model.score(X_train,y_train).round(2))
print('Model Test Score: ',model.score(X_test,y_test).round(2))
#####################################################################