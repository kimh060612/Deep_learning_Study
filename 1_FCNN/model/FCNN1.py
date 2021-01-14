from tensorflow import keras
import numpy as np

class FCNN(keras.Model):
    def __init__(self, _units = [56, 56], _activation = ['relu', 'relu'], **kwargs):
        super().__init__(**kwargs)
        self.Hidden1 = keras.layers.Dense(units=_units[0], activation=_activation[0], kernel_initializer="normal")
        self.Hidden2 = keras.layers.Dense(units=_units[1], activation=_activation[1], kernel_initializer="normal")
        self._output = keras.layers.Dense(units=10, activation="softmax")
    
    def call(self, Input):
        hidden1 = self.Hidden1(Input)
        hidden2 = self.Hidden2(hidden1)
        Output = self._output(hidden2)
        return Output
        
    