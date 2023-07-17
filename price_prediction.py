# Redes neurais
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from itertools import product

new_data = pd.read_csv('kc_house_data.csv')

# Pré-processamento
data = new_data.sample(frac=1)
data = data.drop(['id', 'date', 'zipcode'], axis=1)
data = data.dropna()
X = data.drop('price', axis=1)
y = data['price']

# Treinamento
kf = KFold(n_splits=5, shuffle=True)
param_grid = {'num_layers':[2, 6], 'num_neurons': [16, 64]}
input_size = len(X.columns)
activation_func = 'relu'
num_epochs = 1

best_loss = float('inf')
best_model = None
best_X_train = None
best_y_train = None
best_X_test = None
best_y_test = None

# Iteração dos hiperparâmetros
for params in product(param_grid['num_layers'], param_grid['num_neurons']):
    num_layers, num_neurons = params
    # Montagem do modelo
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_size, activation='linear', input_shape=(input_size,)))
    for _ in range(num_layers-1):
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
        print(f'Fold {fold+1}, {params}, loss: {loss}')
        if loss < best_loss:
            best_loss = loss
            best_model = tf.keras.models.clone_model(model)
            best_X_train = X_train
            best_y_train = y_train
            best_X_test = X_test
            best_y_test = y_test
            best_param = params
    print(f'Média: {np.mean(losses)} ± {np.std(losses)}')

best_predictions = best_model.predict(best_X_test, verbose=0)
print('Best params')
print(best_param)
print('Predictions')
print(best_predictions)
print('Best_y_test')
print(best_y_test)


# Mapas
