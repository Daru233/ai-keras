from keras import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# inputs1 = Input(shape=(3, 1))
# lstm1 = LSTM(1)(inputs1)
# model = Model(inputs=inputs1, outputs=lstm1)
# data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# print(model.predict(data))
