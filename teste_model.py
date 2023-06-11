import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
# Carregar o modelo treinado a partir do arquivo

def etl(df):
    columns = ['gender','smoking_history']
    mapping = {}
    category = ''
    encoding = 0

    for column in columns:
        unique_values = df[column].unique()
        #print('coluna = ',column)
        for i, value in enumerate(unique_values):
            mapping[value] = i
            category = value
            encoding = i

    dfnew = df
    dfnew = dfnew.replace(mapping)
    return dfnew


def fazer_previsao(df):
    modelo_carregado = joblib.load('modelo.pkl')

    dados_preprocessados = etl(df)

    previsao = modelo_carregado.predict(dados_preprocessados)

    print(previsao)

    return previsao

df = pd.read_csv('data/test_dataset.csv',header=0)

df_r = df['diabetes']

df['age'] = df['age'].astype('int')

dfnew = df.drop(columns = ['diabetes'])

acuracia = accuracy_score(df_r, fazer_previsao(dfnew))

print(f'\nAcurácia do modelo: {acuracia*100:.2f}%')
