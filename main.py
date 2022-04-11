from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import math
import matplotlib.pyplot as plt

# load data
twitter_stock_prices_path = 'TWTR.csv'
austria_power_consumption_path = 'at.csv'
train = pd.read_csv(twitter_stock_prices_path)
csv_path = 'jena_climate_2009_2016.csv'

data = pd.read_csv(austria_power_consumption_path)
data_de = data.drop(['end'], axis=1)
print(data_de)