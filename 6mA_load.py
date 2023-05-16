# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:02:49 2023

@author: user
"""
import pandas as pd
from joblib import load
X_tst = pd.read_csv('features/test.csv', header=None)

clf = load('models/H.sapiens.joblib')
print('Prediction result: ', clf.predict(X_tst)[0])