import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
import pickle
import cv2

# Reading the binary files previously created:
file = open("Feature.dat", "rb")
Feature = pickle.load(file)
file.close()
file = open("Label.dat", "rb")
Label = pickle.load(file)
file.close()

Feature = np.array(Feature / 255.0)
Label = np.array(Label)

# Creating the model:

model = Sequential()

# Using 64filters and a kerel size of 3*3 and passing the Feature array:
model.add(Conv2D(64, (3, 3), input_shape=Feature.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, "relu"))
model.add(Dense(1))
model.add(Activation("relu"))

model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # Compiling the model

model.fit(Feature, Label, batch_size=32, epochs=5, validation_split=0.3)  # Training the model

model.save("T-shirt-CNN-64x3.model")  # saving the model