# univariate multi-step encoder-decoder convlstm
from math import sqrt
import tensorflow as tf
import numpy as np
from numpy import split
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.recurrent import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.metrics import RootMeanSquaredError
from keras.optimizers import adam_v2

gb_power_consumption_path = 'be.csv'

# UTC start time
# UTC end time
# power consumption (MW)
df = read_csv(gb_power_consumption_path)
# every sample is every 15 minutes
# I want to take every 1 hr
df = df[::4]
# make index to be of the start date time
df.index = to_datetime(df['start'], format='%Y.%m.%d %H:%M:%S')
# be_load = df['load'] is a series, minmaxscaler only accepts df
be_load = df['load']
# train = df.iloc[:47000]
# test = df.iloc[47000:]
# train = train[['load']]
# test = test[['load']]
# be_load = df[['load']]
#
# scaler = MinMaxScaler()
# scaler.fit(be_load)
# scaled_train = scaler.transform(train)
# scaled_test = scaler.transform(test)
#
# # df_scaled_be_load = pd.DataFrame(scaled_be_load)
# # TODO watch this https://www.youtube.com/watch?v=S8tpSG6Q2H0&t=281s
# # TODO do like iloc: https://github.com/nachi-hebbar/Time-Series-Forecasting-LSTM/blob/main/RNN_Youtube.ipynb
#
# # scaled_train = scaled_be_load[:47000]
# # scaled_test =  scaled_be_load[47000:]
#
# # print(f'Scaled Train Shape {scaled_train.shape}, Scaled Test {scaled_test.shape}')
#
# N_INPUT = 6
# N_FEATURES = 1
# generator = TimeseriesGenerator(scaled_train, scaled_train, length=N_INPUT, batch_size=42)
#
# X, y = generator[0]
# # print(f'Given the Array: {X.flatten()}')
# # print(f'Predict this y: {y}')
# # print(f'X shape: {X.shape}')
#
# model = Sequential()
# model.add(LSTM(128, activation='relu', kernel_initializer='he_normal', input_shape=(N_INPUT, N_FEATURES)))
# # model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
# # model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1, activation='linear'))
# optimzer = adam_v2.Adam(learning_rate=0.0001)
# model.compile(loss='mse', optimizer=optimzer, metrics=[RootMeanSquaredError(), 'accuracy'])
# model.summary()
#
# model.fit(generator, epochs=1)
#
# loss_per_epoch = model.history.history['loss']
# # plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
# # plt.show()
#
#
# test_predictions = []
# first_eval_batch = scaled_train[-N_INPUT:]
# # print(first_eval_batch)
# current_batch = first_eval_batch.reshape((1, N_INPUT, N_FEATURES))
# # print(current_batch)
#
# print('before')
# for i in range(len(scaled_test)):
#     current_pred = model.predict(current_batch)[0]
#     # print(f'current_pred {current_pred}')
#     test_predictions.append(current_pred)
#     # print(f'test_pred {test_predictions}')
#     current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
#
# true_predictions = scaler.inverse_transform(test_predictions)
#
# be_load[47000:] = true_predictions
# true_predictions = pd.DataFrame(true_predictions)
# true_predictions.plot()
# plt.show()
#
# import sys; sys.exit()

def df_to_X_y(df, WINDOW_SIZE):
    # df_as_np = df.to_numpy()
    df_as_np = df
    X = []
    y = []
    for i in range(len(df_as_np) - WINDOW_SIZE):
        row = [[a] for a in df_as_np[i:i + WINDOW_SIZE]]
        # print(f'Row {row}')
        X.append(row)
        label = df_as_np[i + WINDOW_SIZE]
        # print(f'Label {label}')
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 6
EPOCHS = 5
BATCH_SIZE = 32
VERBOSE = 2
POINTS_TO_DISPLAY = 250
N_INPUT = 3
N_FEAWTURES = 1

# generator = TimeseriesGenerator()

X, y = df_to_X_y(be_load, WINDOW_SIZE)
print(f'X Shape: {X.shape}, -- Y Shape: {y.shape}')

X_train, y_train = X[:35000], y[:35000]
X_val, y_val = X[35000:40000], y[35000:40000]
X_test, y_test = X[40000:], y[40000:]

print(f'{X_train.shape}, {y_train.shape}, {X_val.shape}, {y_val.shape}, {X_test.shape}, {y_test.shape} ')

model = Sequential()
model.add(LSTM(128, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='linear'))
model.summary()

optimzer = adam_v2.Adam(learning_rate=0.0003)
# checkpoint = ModelCheckpoint('model/', save_best_only=True)
model.compile(loss='mse', optimizer=optimzer, metrics=[RootMeanSquaredError(), 'accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
fig, axs = plt.subplots(3)
fig.suptitle('Train, Validation, Test')

# mse, mae = model.evaluate(X_test, y_test, verbose=0)
# print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))

train_predictions = model.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
# plt.plot(train_results['Train Predictions'][:100])
# plt.plot(train_results['Actuals'][:100])
# plt.show()
axs[0].set_title('Train Predictions')
axs[0].plot(train_results['Train Predictions'][:POINTS_TO_DISPLAY], label='Predictions')
axs[0].plot(train_results['Actuals'][:POINTS_TO_DISPLAY], label='Actuals')
axs[0].legend(loc='upper left')

val_predictions = model.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions': val_predictions, 'Actuals': y_val})
# plt.plot(val_results['Val Predictions'][:100])
# plt.plot(val_results['Actuals'][:100])
# plt.show()
axs[1].set_title('Val predictions')
axs[1].plot(val_results['Val Predictions'][:POINTS_TO_DISPLAY], label='Predictions')
axs[1].plot(val_results['Actuals'][:POINTS_TO_DISPLAY], label='Actuals')
axs[1].legend(loc='upper left')

test_predictions = model.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})
# plt.plot(test_results['Test Predictions'][:100])
# plt.plot(test_results['Actuals'][:100])
# plt.show()
axs[2].set_title('Test Predictions')
axs[2].plot(test_results['Test Predictions'][:POINTS_TO_DISPLAY], label='Predictions')
axs[2].plot(test_results['Actuals'][:POINTS_TO_DISPLAY], label='Actuals')
axs[2].legend(loc='upper left')

plt.show()
