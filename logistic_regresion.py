#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: codaxor
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('appended_songs.csv',index_col=0)

X = data.iloc[:,4:].values
y = data.iloc[:,2].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)