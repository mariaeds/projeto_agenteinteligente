import pandas as pd
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


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
            #print('categoria = ',category, '(',encoding,')')

    dfnew = df
    dfnew = dfnew.replace(mapping)
    return dfnew

def fazer_previsao(df):
    modelo_carregado = joblib.load('modelo.pkl')

    dados_preprocessados = etl(df)

    previsao = modelo_carregado.predict(dados_preprocessados)

    #print(previsao)

    return previsao

#QUESTIONARIO
gender = input("Digite seu gênero (homem, mulher, outro): ")
age = input("Digite sua idade: ")
hypertension = input("Possui hipertensão (0 - Não, 1 - Sim): ")
heart_disease = input("Possui doença cardíaca (0 - Não, 1 - Sim): ")
smoking_history = input("É fumante? \n(atualmente nao - atualmente sim - ja fumei - NI - nunca - sempre):\n ")
bmi = input("Indice de massa corporal (IMC): ")
HbA1c_level = input("Nível de hemoglobina glicada: ")
blood_glucose_level = input("Média do nível de glicose no sangue: ")


# df = pd.DataFrame([[1,1.23,'Hello']], columns=list('ABC'))
#              
d = {'gender': [gender], 'age': [age],'hypertension':hypertension,'heart_disease':heart_disease,
'smoking_history':smoking_history,
     'bmi':bmi,'HbA1c_level':HbA1c_level,'blood_glucose_level':blood_glucose_level}

df = pd.DataFrame([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]],
                 columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level',
                          'blood_glucose_level'] )

#print(df)

resul = fazer_previsao(df)
if resul[0] == 0:
    print(f'Resultado: Negativo')
else:
    print(f'Resultado: Positivo')
