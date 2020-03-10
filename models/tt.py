import pandas as pd
import numpy as np
import sys
import os
import random
import xgboost

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from statsmodels import robust
from scipy import stats

from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


#duvida SLC15, ASRI42, ASRI_pass42 (pensava em morrer ou em suicidio)

# Parameters
LABEL_COLUMN_NAME = 'bug'
#UNWANTED_COLUMNS = []
UNWANTED_COLUMNS = ['name']

N_FOLDS = 5
RANDOM_STATE = 1

def eval_bootstrap(df, features, md):
    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values

    aa = []
    bb = []
    cc = []
    dd = []
    for i in range(1,5):
        a = []
        b = []
        c = []
        d = []
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=i)
        for (train, val) in cv.split(X, y):
            if md == 1: regressor = ensemble.GradientBoostingRegressor(n_estimators = 30, max_depth = 4, min_samples_split = 2, learning_rate = 0.1, loss = 'ls', random_state = RANDOM_STATE)
            elif md == 2: regressor = ensemble.RandomForestRegressor(n_estimators = 30, max_depth = 10, min_samples_split = 4, random_state = RANDOM_STATE)
            elif md == 3: regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
            elif md == 4: regressor = MLPRegressor(hidden_layer_sizes=(20,30,30,5,), batch_size = 10, activation='relu', random_state = RANDOM_STATE)
            elif md == 5: regressor = LinearRegression()
            elif md == 6: regressor = Lasso(alpha=0.1, random_state = RANDOM_STATE)
            elif md == 7: regressor = xgboost.XGBClassifier(max_depth=7, n_estimators=200)           

            regressor = regressor.fit(X[train], y[train])
            pred = regressor.predict(X[val])

            predictions = [round(value) for value in pred]

            accuracy = accuracy_score(y[val], predictions)

            #rmse = np.sqrt(np.mean((pred - y[val])**2))
            #mae = mean_absolute_error(pred, y[val])
            #r2 = r2_score(pred, y[val])
            #results = cross_val_score(regressor, X[val], y[val], cv=cv, scoring='roc_auc')

        aa.append(accuracy*100)
        bb.append(1)
        cc.append(1)
    return np.mean(aa),np.mean(bb),np.mean(cc)

def back_one(df, f, md):
    v = 0
    f1 = []
    f2 = []
    for i in f:
        f1.insert(len(f1), i)
        f2.insert(len(f2), i)
    A,B,C = eval_bootstrap(df, f1, md)
    z = A
    for i in f:
        f1.remove(i)
        A,B,C = eval_bootstrap(df, f1, md)
        #print("%s,%f,%f,%f" % (f1,A,B,C))
        if A < z:
            v = 1
            z = A
            f2 = []
            for j in f1:
                f2.insert(len(f2), j)
        f1.insert(len(f1), i)
    return v,f2

# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

RANDOM_STATE = 1
all_features = list(df.columns)

#f = []
#for x in UNWANTED_COLUMNS:
#    if x in all_features: f.insert(len(f), x)
#for x in f + [LABEL_COLUMN_NAME]:
for x in UNWANTED_COLUMNS + [LABEL_COLUMN_NAME]:
    all_features.remove(x)

md = int(sys.argv[2])
f = []
i = 0
max = 0
c = 0
total = 0
for f1 in all_features:
    if i == 5: break
    if f1 in f: continue
    k = 1000
    x = f1
    i = i + 1
    j = 0

    for f2 in all_features:
         if f2 in f: continue
         j = j + 1
         f.insert(len(f), f2)
         A,B,C = eval_bootstrap(df, f, md)
         
         total = total + 1
         if A > 79.5:
             c = c + 1

         if A > max:
             max = A
         print("%s,%f,%f" % (f,A, max))
         z = A
         f.remove(f2)
         sys.stdout.flush()
         if z < k:
             x = f2
             k = z
    f.insert(len(f), x)
    if i > 2:
         v,f = back_one(df, f, md)
         while v == 1:
             v,f = back_one(df, f, md)
         i = len(f)

print(c/total)
