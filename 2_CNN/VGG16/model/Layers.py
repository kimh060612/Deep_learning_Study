from tensorflow import keras
import tensorflow as tf

class CustomConvLayer2(keras.layers.Layer):
    def __init__(self, num_channels, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.Conv1 = keras.layers.Conv2D(num_channels, kernel_size=(3,3), padding="SAME", activation='relu')
        self.Conv2 = keras.layers.Conv2D(num_channels, kernel_size=(3,3), padding="SAME", activation='relu')
        self.MaxPooling1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

    def call(self, Input):
        X = self.Conv1(Input)
        X = self.Conv2(X)
        X = self.MaxPooling1(X)
        return X


class CustomConvLayer3(keras.layers.Layer):
    def __init__(self, num_channels, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.Conv1 = keras.layers.Conv2D(num_channels, kernel_size=(3,3), padding="SAME", activation='relu')
        self.Conv2 = keras.layers.Conv2D(num_channels, kernel_size=(3,3), padding="SAME", activation='relu')
        self.Conv3 = keras.layers.Conv2D(num_channels, kernel_size=(3,3), padding="SAME", activation='relu')
        self.MaxPooling1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

    def call(self, Input):
        X = self.Conv1(Input)
        X = self.Conv2(X)
        X = self.Conv3(X)
        X = self.MaxPooling1(X)
        return X

