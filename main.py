# univariate multi-step encoder-decoder convlstm
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


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    # total rows 1443
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):

    print(f'actual {actual[:5]}   predicted: {predicted[:5]}')
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores, loss_per_epoch):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].set_title('Loss per epoch')
    axs[0].plot(loss_per_epoch, marker='o', label='loss')
    axs[0].legend(loc='upper left')
    axs[0].set_label('Loss')

    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    axs[1].set_title('RMSE score per day')
    axs[1].plot(days, scores, marker='o', label='lstm')
    axs[1].legend(loc='upper left')

    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))

    plt.show()


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1

    return array(X), array(y)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    VERBOSE = 2
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.008
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    OPTIMZER = adam_v2.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse', optimizer=OPTIMZER, metrics=['mae', 'mse'])

    # fit network
    history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    loss_per_epoch = history.history['loss']
    return model, loss_per_epoch


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data

    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))

    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model, loss_per_epoch = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores, loss_per_epoch


def main():
    # load the new file
    dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                       parse_dates=['datetime'], index_col=['datetime'])

    # split into train and test
    train, test = split_dataset(dataset.values)
    # evaluate model and get scores
    n_input = 7
    score, scores, loss_per_epoch = evaluate_model(train, test, n_input)
    # summarize scores
    summarize_scores('lstm', score, scores, loss_per_epoch)


if __name__ == '__main__':
    main()
