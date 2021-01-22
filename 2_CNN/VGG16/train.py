from model.model import VGG16
from tensorflow import keras
import tensorflow as tf
import numpy as np

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

CIFAR10 = keras.datasets.cifar10

(train_img, train_labels), (test_img, test_labels) = CIFAR10.load_data()

train_img, test_img = train_img.reshape([-1, 32, 32, 3]), test_img.reshape([-1, 32, 32, 3])

train_img = train_img.astype(np.float32) / 255.
test_img = test_img.astype(np.float32) / 255.

print (train_labels.shape)
print (test_labels.shape)

Optimizer = keras.optimizers.SGD(learning_rate=0.001)
vgg = VGG16()

vgg.compile(optimizer=Optimizer, loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy()])
vgg.fit(train_img, train_labels, batch_size = 64, epochs = 20, verbose = 1, validation_data=(test_img, test_labels))

