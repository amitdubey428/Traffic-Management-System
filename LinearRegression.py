# -*- coding: utf-8 -*-
"""
Description
@file LinearRegression.py
This file is used for training the model over the given values
of the density v/s time. The prediction is done through the 
Linear Regression technique

"""
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Linear.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#implementing our classifier based 
from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(X_train,y_train)

y_predict=simplelinearRegression.predict(X_test)
y_predict_val=simplelinearRegression.predict(5.5)
 #graph
 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,simplelinearRegression.predict(X_train))
plt.show()
