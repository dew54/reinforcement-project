import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import keras
import csv
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
import os





#y_commands = pd.read_csv('commands.csv')
key_vector = pd.read_csv('key_vector.csv')
x_img = pd.read_csv('img.csv')
#x_cardata= pd.read_csv('cardata.csv')
print(x_img.shape)
#print(x_cardata.shape)
#print(y_commands.shape)
print(x_img.ndim)
x_img=x_img.to_numpy()
#x_cardata=x_cardata.to_numpy()

x_img= np.reshape(x_img, (-1, 160, 90))
#print(x_img.shape)

def listify(*args):
  return list(args)


#xx_train = listify(x_img, x_cardata)

x_data = np.array([x_img])
k_data = np.array([key_vector, key_vector])

#print("xxtrainshape:", xx_train.__len__())


#print("xxtrainshape:", yy_train.shape)

print(x_img.shape)

x_val = x_data[-974:]
y_val = k_data[-974:]
x_train = x_data[:-3000]
y_train = k_data[:-3000]


print('Data created successfully')
print(tf.config.list_physical_devices('GPU'))

gpu = tf.config.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)



# Create the model
model = keras.Sequential()
model.add(tf.keras.Input(shape=(160, 90, 1)))
model.add(keras.layers.convolutional.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dense(units = 64, activation = 'linear'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.convolutional.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units = 64, activation = 'softmax'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 64, activation = 'tanh'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 5, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")

# Display the modelt
model.summary()

model.fit( x_train, y_train, epochs=500, verbose=1, batch_size=64, validation_data=(x_val, y_val),)

#y_predicted = model.predict(x_data)

# save the result
#model.save_weights('./checkpoints/my_checkpointval1')
#model.save('saved_model/my_modelval1')
#model.save('my_modelval1.h5')

