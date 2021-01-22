from tensorflow import keras
from model.Layers import CustomConvLayer2, CustomConvLayer3
import tensorflow as tf

class VGG16(keras.Model):
    def __init__(self):
        super().__init__()
        self.ConvBlock1 = CustomConvLayer2(64)
        self.ConvBlock2 = CustomConvLayer2(128)
        self.ConvBlock3 = CustomConvLayer3(256)
        self.ConvBlock4 = CustomConvLayer3(512)
        self.ConvBlock5 = CustomConvLayer3(512)
        self.Flat = keras.layers.Flatten()
        self.Dense1 = keras.layers.Dense(4096, activation='relu')
        self.Dense2 = keras.layers.Dense(4096, activation='relu')
        self.Dense3 = keras.layers.Dense(1000, activation='softmax')

    def call(self, Input):
        X = self.ConvBlock1(Input)
        X = self.ConvBlock2(X)
        X = self.ConvBlock3(X)
        X = self.ConvBlock4(X)
        X = self.ConvBlock5(X)
        X = self.Flat(X)
        X = self.Dense1(X)
        X = self.Dense2(X)
        X = self.Dense3(X)
        return X