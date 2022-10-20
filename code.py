# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:37:56 2022

@author: a2452
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

print(data.head())

print(data.isnull().any()) 

#--------------------------------------------------------------

#heatmap
plt.figure(figsize=(15, 12))
feature_corr = data.corr()
hm = sns.heatmap(feature_corr, annot=True)
plt.show()

#--------------------------------------------------------------

X = data[["age", "time"]].values
y = data[["DEATH_EVENT"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=345453)


#training 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

#score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#--------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.scatter(data["time"],data["age"] ,s = 60 , c=y)

def plot_decision_boundary(m):

    #set max value, min value and edge filling
    x_min  = data["time"].min() - .5
    x_max  = data["time"].max() + .5

    y_min = data["age"].min() - .5
    y_max = data["age"].max() + .5
    
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #use prediction function to predict
    Z = m(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #plot
    plt.contourf(xx, yy, Z)

    for i in range(data.shape[0]):
        if y[i] == 1:
            died = plt.scatter(data.time[i],data.age[i],marker = "x", s = 40 , color = "green")
        
        else:
            alive = plt.scatter(data.time[i],data.age[i],marker = "o", s = 40 , color = "blue")
    
    plt.legend((died,alive),('1','0'),title= "DEATH_EVENT")


plot_decision_boundary(lambda x: classifier.predict(x))
plt.xlabel('Time')  
plt.ylabel('Age') 
plt.title("Logistic Regression")
plt.show()

