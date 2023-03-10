# -*- coding: utf-8 -*-
"""
@author: Group 6
"""
#Imports for data, cf : https://keras.io/datasets/
from keras.datasets import mnist
#Imports for CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical

import numpy as np

from os import system
import cnnutils_v2 as cu

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data("/chemin/absolu/vers/mnist.npz")

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy as a measure of model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
a = system("clear")
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=10)

# Evaluate the model
cu.print_model_error_rate(model, X_test, y_test)
# Save the model
cu.save_keras_model(model, "save_model_v2/no_preparation_small_model_v2_cnn")
