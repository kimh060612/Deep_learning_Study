from tensorflow import keras
import tensorflow as tf

class AlexNet(keras.Model):
    def __init__(self):
        super().__init__()
        # 원래 여기에는 커널 사이즈로 (11, 11)이 들어가고 padding은 valid이다. 하지만 메모리 때문에 돌아가지 않는 관계로 이미지 크기를 줄아느라 부득이하게 모델을 조금 변경했다.
        self.Conv1 = keras.layers.Conv2D(96, (3, 3), strides=(4, 4), padding="SAME", activation="relu")
        # LRN 1
        self.BatchNorm1 = keras.layers.BatchNormalization()
        self.MaxPool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="VALID")

        self.Conv2 = keras.layers.Conv2D(256, kernel_size=(5, 5), padding="SAME", activation="relu")
        # LRN2
        self.BatchNorm2 = keras.layers.BatchNormalization()
        self.MaxPool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="VALID")

        self.Conv3 = keras.layers.Conv2D(384, kernel_size=(3, 3), padding="SAME", activation="relu")
        self.Conv4 = keras.layers.Conv2D(384, kernel_size=(3, 3), padding="SAME", activation="relu")
        self.Conv5 = keras.layers.Conv2D(256, kernel_size=(3, 3), padding="SAME", activation="relu")
        self.MaxPool3 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        self.Flat = keras.layers.Flatten()

        self.Dense1 = keras.layers.Dense(4096, activation="relu")
        self.DropOut1 = keras.layers.Dropout(0.5)

        self.Dense2 = keras.layers.Dense(4096, activation="relu")
        self.DropOut2 = keras.layers.Dropout(0.5)

        self.OutDense = keras.layers.Dense(10, activation="softmax")


    def call(self, Input):
        X = self.Conv1(Input)
        #X = tf.nn.local_response_normalization(X)
        X = self.BatchNorm1(X)
        X = self.MaxPool2(X)
        X = self.Conv2(X)
        #X = tf.nn.local_response_normalization(X)
        X = self.BatchNorm2(X)
        X = self.MaxPool2(X)
        X = self.Conv3(X)
        X = self.Conv4(X)
        X = self.Conv5(X)
        X = self.MaxPool3(X)
        X = self.Flat(X)
        X = self.Dense1(X)
        X = self.DropOut1(X)
        X = self.Dense2(X)
        X = self.DropOut2(X)
        X = self.OutDense(X)
        return X