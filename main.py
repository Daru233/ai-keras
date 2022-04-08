from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

# load data
twitter_stock_prices_path = 'TWTR.csv'
train = pd.read_csv(twitter_stock_prices_path)

csv_path = 'jena_climate_2009_2016.csv'

df = pd.read_csv(csv_path)

df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

close = df['T (degC)']
close.plot()

def df_to_X_y(df, window_size):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)


WINDOW_SIZE = 5
X, y = df_to_X_y(close, WINDOW_SIZE)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(InputLayer((5, 1)))
model.add(LSTM(128))
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))
model.summary()

cp = ModelCheckpoint('model/', save_best_only=True)
optimzer = adam_v2.Adam(learning_rate=0.0001)
model.compile(loss=MeanSquaredError(), optimizer=optimzer, metrics=[RootMeanSquaredError()])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[cp])

model = load_model('model/')

train_predictions = model.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actual': y_train})
print(train_results)