from math import sqrt
import tensorflow as tf
import numpy as np
from numpy import split, nan
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.metrics import RootMeanSquaredError
from keras.optimizers import adam_v2


def split_sequence(sequence, n_input):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_input
        # check if we are beyond the sequence
        if end_ix > len(sequence) - n_input:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix + n_input]
        X.append(seq_x)
        y.append(seq_y)
    return np.asarray(X), np.asarray(y)


def define_model(X_train, y_train, X_val, y_val):
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    # reshape output into [samples, timesteps, features]
    train_y_reshaped = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))

    VERBOSE = 2
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0008
    OPTIMZER = adam_v2.Adam(learning_rate=LEARNING_RATE)

    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(200, activation='relu')))
    model.add(TimeDistributed(Dense(12, activation='linear')))
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='mse', optimizer=OPTIMZER, metrics=['mae', RootMeanSquaredError()])
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        verbose=VERBOSE)

    score = model.evaluate(X_train, y_train)
    print('Evaluation Scores')
    print(f' MSE: {score[0]:.3f}, MAE: {score[1]:.3f}, RMSE: {score[2]}')

    return model, history


def main():

    be_electricity_path = 'be.csv'
    df = read_csv(be_electricity_path)
    # every sample is 15 minutes, I want to take every 4hr
    df = df[::16]
    # make index to be of the start date time
    df.index = to_datetime(df['start'], format='%Y.%m.%d %H:%M:%S')
    be_load = df['load']
    be_load_values = be_load.values.astype('float32')
    print(f'len of be_load {len(be_load_values)}')

    # 12 hours worth of 'time'
    n_input = 3

    X, y = split_sequence(be_load_values, n_input)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # rough split of 70% train, 15% validation, 15% test
    X_train, y_train = X[:8563], y[:8563]
    # 15% of data
    X_val, y_val = X[8563:10398], y[8563:10398]
    # 15% of data
    X_test, y_test = X[10398:], y[10398:]

    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'X_val: {X_val.shape}, y_val: {y_val.shape}')
    print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

    model, history = define_model(X_train, y_train, X_val, y_val)

    loss = history.history['loss']
    mae = history.history['mae']
    root_mean_squared_error = history.history['root_mean_squared_error']
    val_loss = history.history['val_loss']
    val_mae = history.history['val_mae']
    val_root_mean_squared_error = history.history['val_root_mean_squared_error']

    fig, axs = plt.subplots(4)
    fig.suptitle('loss, val_loss')
    axs[0].set_title('loss, val_loss')
    axs[0].plot(loss, marker='.', label='loss')
    axs[0].plot(val_loss, marker='.', label='val_loss')
    axs[0].legend(loc='upper center')

    axs[1].set_title('mae, val_mae Loss')
    axs[1].plot(mae, marker='.', label='mae')
    axs[1].plot(val_mae, marker='.', label='val_mae')
    axs[1].legend(loc='upper center')

    axs[2].set_title('rmse, val_rmse')
    axs[2].plot(root_mean_squared_error, marker='.', label='root_mean_squared_error')
    axs[2].plot(val_root_mean_squared_error, marker='.', label='val_root_mean_squared_error')
    axs[2].legend(loc='upper center')



    test_predictions = model.predict(X_test)
    predictions_plot = []
    actual_plot = []
    for i in range(len(test_predictions)):
        predictions_plot.append(test_predictions[i][0].flatten())
        actual_plot.append(y_test[i][0].flatten())
        print(f'Predictions: {test_predictions[i][0].flatten()}, Actuals: {y_test[i][0].flatten()}')

    POINT_TO_DISPLAY = 200

    axs[3].set_title('predicted, actual')
    axs[3].plot(predictions_plot[:POINT_TO_DISPLAY], marker='.', label='predictions')
    axs[3].plot(actual_plot[:POINT_TO_DISPLAY], marker='.', label='actual')
    axs[3].legend(loc='upper center')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.show()


if __name__ == '__main__':
    main()
