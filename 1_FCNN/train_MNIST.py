from model.FCNN1 import FCNN
from tensorflow import keras
import tensorflow as tf
import numpy as np

mnist = keras.datasets.mnist
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

train_img, test_img = train_img.reshape([-1, 784]), test_img.reshape([-1, 784])

train_img = train_img.astype(np.float32) / 255.
test_img = test_img.astype(np.float32) / 255.

model = FCNN()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.fit(train_img, train_labels, batch_size = 32, epochs = 15, verbose = 1, validation_split = 0.3)


model.evaluate(test_img, test_labels, verbose=2)
