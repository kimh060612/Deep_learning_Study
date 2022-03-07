import tensorflow as tf
import numpy as np
from tensorflow import keras
from torch import dropout

class RNNLayer(keras.Model):
    def __init__(self, num_hidden=128, num_class=39):
        super(RNNLayer, self).__init__()
        self.RNN1 = keras.layers.SimpleRNN(num_hidden)
        self.RNN2 = keras.layers.SimpleRNN(num_hidden)
        self.out = keras.layers.Dense(num_class, activation="softmax")
        self.out = keras.layers.TimeDistributed(self.out)
        
    def call(self, x):
        x = self.RNN1(x)
        x = self.RNN2(x)
        out = self.out(x)
        return out