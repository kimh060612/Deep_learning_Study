from model.ConvModel import ConvModel
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

## Data Preprocessing
mnist = keras.datasets.mnist
# 28*28 image ==> 1*(28*28) ==> 1*784
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

train_img, test_img = train_img.reshape([-1, 28, 28, 1]), test_img.reshape([-1, 28, 28, 1])

train_img = train_img.astype(np.float32) / 255.
test_img = test_img.astype(np.float32) / 255.

print (train_labels.shape)
print (test_labels.shape)

CNN_model = ConvModel()
CNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy()])
# 모델을 학습시키는 부분.
CNN_model.fit(train_img, train_labels, batch_size = 32, epochs = 15, verbose = 1, validation_split = 0.3)
#                                  1 배치의 크기                                    Validation Dataset의 크기. 60000*0.3 = 18000

CNN_model.evaluate(test_img, test_labels, verbose=2)