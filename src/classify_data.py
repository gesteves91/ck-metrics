import pandas as pd
import pickle
import xgboost
import re
import joblib

from sklearn import model_selection
from pandas import DataFrame

x = pd.read_csv('~/sources/ck-metrics/experiments/total.csv')

y = pd.read_csv('~/sources/ck-metrics/experiments/data-processed.csv')
y = y[['feature']].copy()

list_features = y.values

df = x[x['bug'] == 1]

list = []
result = []

results = []

def classify_dataset(models):
    for (features, model) in zip(list_features, models):
        df_new = pd.DataFrame()
        my_string = str(features)
        my_list = my_string.split(",")
        for feature in my_list:
            new_f = re.sub('[^a-zA-Z0-9 \n\.]', '', feature)
            new_f = new_f.replace(" ", "")
            if new_f == "maxcc":
                new_f = "max_cc"
            if new_f == "avgcc":
                new_f = "avg_cc"
            df_new[new_f] = df[new_f]
        df_new = rename_columns(df_new)
        df_new['bug'] = df['bug']
        X_test = getX(df_new)
        result = model.predict(X_test)
        results.append(result)
    return results

def rename_columns(df_rename):
    length = len(df_rename.columns)
    list_names = []
    index = 0

    for l in range(0,length):
        list_names.append('f%s' % (index))
        index = index + 1

    df_rename.columns = list_names
    return df_rename

def getX(df_feat):
    seed = 1
    cols = [col for col in df_feat.columns if col not in ['bug']]
    X = df_feat[cols]
    Y = df_feat['bug']
    validation_size = 0.2
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)
    return X_test

def transform_list_df(lists):
    list_column_name = [] 

    for index in range(0,2452):
        list_column_name.append('model%s' % (index))
    
    df_final = DataFrame(lists,columns=list_column_name)
    return df_final

loaded_model = joblib.load('/home/geanderson/sources/ck-metrics/data/models.pkl')
list_final = classify_dataset(loaded_model)
df_final = transform_list_df(list_final)
print(df_final.shape)

# save this to a new_df
df_final.to_csv('/home/geanderson/sources/ck-metrics/experiments/model-prediction.csv', index=False, header=True)
