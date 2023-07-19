# Redes neurais
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

new_data = pd.read_csv('kc_house_data.csv')

# Pré-processamento
data = new_data.sample(frac=1)
data = data.drop(['id', 'date', 'zipcode'], axis=1)
data = data.dropna()

scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

X = data.drop('price', axis=1)
y = data['price']

# Treinamento
kf = KFold(n_splits=3, shuffle=True)
layers_sequences = [[32, 32, 32], [32, 64, 32], [64, 64, 64]]
input_size = len(X.columns)
activation_func = 'relu'
num_epochs = 100
best_loss = float('inf')
best_model, best_X_train, best_y_train, best_X_test, best_y_test = None, None, None, None, None

# Iteração dos hiperparâmetros
for layer_sequence in layers_sequences:
    # Montagem do modelo
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_size, activation='linear', input_shape=(input_size,)))
    for num_neurons in layer_sequence:
        model.add(tf.keras.layers.Dense(num_neurons, activation=activation_func))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.Huber()
    model.compile(optimizer=optimizer, loss=loss)

    losses = list()
    # Iteração dos folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, epochs=num_epochs, verbose=0)

        loss = model.evaluate(X_test, y_test, verbose=0)
        losses.append(loss)
        print(f'Fold {fold+1}, {layer_sequence}, loss: {loss}')
        if loss < best_loss:
            best_loss = loss
            best_model = tf.keras.models.clone_model(model)
            best_X_train = X_train
            best_y_train = y_train
            best_X_test = X_test
            best_y_test = y_test
            best_layer_sequence = layer_sequence
    print(f'Média: {np.mean(losses)} ± {np.std(losses)}')

best_predictions = best_model.predict(best_X_test, verbose=0)
print('Best layer sequence:', best_layer_sequence)


# Mapas
lim_pontos = float('inf')
num_pontos = min(data.shape[0], lim_pontos)

