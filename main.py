import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

df = pd.read_csv('diabetes_prediction_dataset.csv')

columns = ['genero','fumante']

for column in columns:
    print('Valores únicos em',column,'=',df[column].nunique())

mapping = {}
category = ''
encoding = 0

for column in columns:
    unique_values = df[column].unique()
    print('coluna = ',column)
    for i, value in enumerate(unique_values):
        mapping[value] = i
        category = value
        encoding = i
        print('categoria = ',category, '(',encoding,')')

dfnew = df
dfnew = dfnew.replace(mapping)


#x_train,x_test,y_train,y_test = train_test_split(dfnew.drop(columns = ['diabetes'] , axis = 1),dfnew['diabetes'],test_size = 0.3)

tf1_lr = ColumnTransformer([('Standard Scaler', StandardScaler(), slice(0,8))])

svm = SVC(kernel='linear')

pipe_svm = Pipeline([('StandardScaler',tf1_lr),('Support Vector Machine',svm)])
pipe_svm.fit(dfnew.drop(columns = ['diabetes'] , axis = 1),dfnew['diabetes'])

joblib.dump(pipe_svm, 'modelo.pkl')