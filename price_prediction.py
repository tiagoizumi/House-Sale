# Redes neurais
import pandas as pd

new_data = pd.read_csv('kc_house_data.csv')

# Pré-processamento
data = new_data.sample(frac=1)
data = data.drop(['id', 'date', 'zipcode'], axis=1)
data = data.dropna()
X_data = data.drop('price', axis=1)
y_data = data['price']



# Mapas
