from model.FCNN1 import FCNN
from tensorflow import keras
import tensorflow as tf
import numpy as np

## Data Preprocessing
mnist = keras.datasets.mnist
# 28*28 image ==> 1*(28*28) ==> 1*784
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# 60000 * 28 * 28  ==> 60000*784

train_img, test_img = train_img.reshape([-1, 784]), test_img.reshape([-1, 784])

train_img = train_img.astype(np.float32) / 255.
test_img = test_img.astype(np.float32) / 255.
###

model = FCNN()
# 모델 확인
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
# 모델을 학습시키는 부분.
model.fit(train_img, train_labels, batch_size = 32, epochs = 15, verbose = 1, validation_split = 0.3)
#                                  1 배치의 크기                                    Validation Dataset의 크기. 60000*0.3 = 18000

model.evaluate(test_img, test_labels, verbose=2)
