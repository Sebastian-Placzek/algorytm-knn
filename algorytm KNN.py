# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:37:32 2019

@author: Seba
"""

import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('KNN_Project_Data')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis =1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

#train test split
from sklearn.model_selection import train_test_split
X= df_feat
y= df['TARGET CLASS']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101 )
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# wybranie odpowiedniego k (minimalizujacego error-value ale też przy relatywnie niskim k)
error_rate= []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='b',linestyle='dashed',marker='o',
         markerfacecolor = 'red', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')