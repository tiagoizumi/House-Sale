# Redes neurais
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


original_data = pd.read_csv('kc_house_data.csv')

# Pré-processamento
original_data = original_data.sample(frac=1)
data = original_data.drop(['id', 'zipcode', 'lat', 'long'], axis=1)
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].apply(lambda date: date.year)
data['month'] = data['date'].apply(lambda date: date.month)
data = data.drop('date', axis=1)
data = data.dropna()

X = data.drop('price', axis=1).values
y = data['price'].values

# Treinamento
kf = KFold(n_splits=3, shuffle=True)
layers_sequences = [[32, 32, 32]]
input_size = X.shape[1]
best_loss = float('inf')
best_model, best_X_train, best_y_train, best_X_test, best_y_test = None, None, None, None, None

# Iteração dos hiperparâmetros
for layer_sequence in layers_sequences:
    # Montagem do modelo
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_size, activation='linear', input_shape=(input_size,)))
    for num_neurons in layer_sequence:
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    losses = list()
    # Iteração dos folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=128)
        loss = model.evaluate(X_test, y_test, verbose=0)
        losses.append(loss)
        print(f'Fold {fold+1}, {layer_sequence}, loss: {loss}')
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)
            best_train_idx = train_index
            best_test_idx = test_index
            best_layer_sequence = layer_sequence
            best_scaler = scaler
    print(f'Média: {np.mean(losses)} ± {np.std(losses)}')

test_pred = best_model.predict(X_test, verbose=0)
plt.scatter(y_test, test_pred)
plt.plot(y_test,y_test, 'r')
plt.show()

predictions = best_model.predict(X, verbose=0)
new_data = data.copy()
new_data['is_test'] = [idx in best_test_idx for idx, _ in new_data.iterrows()]
new_data['price_prediction'] = predictions
# Mapas
lim_pontos = float('inf')
num_pontos = min(data.shape[0], lim_pontos)

data['id'], data['date'] = original_data['id'], original_data['date']
data['lat'], data['long'] =  original_data['lat'], original_data['long']
ordem_das_colunas = ['price', 'NEW PRICE', 'id', 'date', 'bedrooms', 'bathrooms', 'sqft_living',
                     'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                     'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                     'lat', 'long', 'sqft_living15', 'sqft_lot15' ]
data = data.reindex(columns=ordem_das_colunas)
diff_real = data['price']-data['NEW PRICE']
data['diferenca'] = diff_real

lat_ini = np.mean([data.lat.min(),data.lat.max()])
long_ini = np.mean([data.long.min(),data.long.max()])
mapObj = folium.Map(location=[lat_ini,long_ini], zoom_control=False)

# Marcadores
cont = 0
limite1 = data[:num_pontos+1]['diferenca'].min()
limite2 = 0
limite3 = data[:num_pontos+1]['diferenca'].max()

colormap = cm.LinearColormap(colors=['green', 'white', 'red'], index=[limite1, limite2, limite3], vmin=limite1, vmax=limite3)

for index,loc in data.iterrows():
    # tip = f"${loc.price}"
    tip = str()
    cont += 1
    if cont > num_pontos:
        break
    for col, value in loc.items():
        tip += f"\t{col}: {value}<br>"
    folium.CircleMarker(location=[loc.lat,loc.long],radius=3,
                        tooltip=tip, fill=True, fill_color=colormap(loc['diferenca']),
                        fill_opacity=1, weight=1, color='black').add_to(mapObj)

mapObj.save('output.html')