# Importar as bibliotecas necessárias
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Carregar o arquivo CSV
file_path = '/mnt/data/weather.nominal.csv'
data = pd.read_csv(file_path)

# Codificação de variáveis categóricas usando LabelEncoder
label_encoders = {}
for column in data.columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Separar variáveis de entrada (X) e alvo (y)
X = data[['outlook', 'temperature', 'humidity', 'windy']]
y = data['play']

# Treinar o modelo Naive Bayes categórico
model = CategoricalNB()
model.fit(X, y)

# Registrar as novas observações:
# Aparência = Chuva (rainy), Temperatura = Fria (cool), Umidade = Normal (normal), Ventando = Sim (True)
new_data = [[label_encoders['outlook'].transform(['rainy'])[0],
             label_encoders['temperature'].transform(['cool'])[0],
             label_encoders['humidity'].transform(['normal'])[0],
             label_encoders['windy'].transform([True])[0]]]

# Calcular as probabilidades de "Jogar" ou "Não Jogar"
probabilities = model.predict_proba(new_data)

# Exibir as probabilidades
print(probabilities)
