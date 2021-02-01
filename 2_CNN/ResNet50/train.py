import tensorflow as tf
from tensorflow import keras
from model.model import ResNet50
import numpy as np
import os
import cv2 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

local_dir = os.path.abspath("/data/KHU_Deep_learning_Study/2_CNN")

train_img = np.load(local_dir + "/train_Image.npy")
train_label = np.load(local_dir + "/train_label.npy")
test_img = np.load(local_dir + "/test_Image.npy")
test_label = np.load(local_dir + "/test_label.npy")

"""print(train_img.shape)
print(train_label.shape)
print(test_img.shape)
print(test_label.shape)"""

Optimizer = keras.optimizers.Adam(learning_rate=0.001)
res = ResNet50()

res.compile(optimizer=Optimizer, loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy()])
res.fit(train_img, train_label, batch_size = 1, epochs = 20, verbose = 1, validation_data=(test_img, test_label))