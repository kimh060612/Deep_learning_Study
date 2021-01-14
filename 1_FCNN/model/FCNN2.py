import tensorflow as tf
from tensorflow import keras
from model.layer import WeightSum, WeightSumBuild


class FCNN(keras.layers.Layer):
    def __init__(self, _units = [56, 56],trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.Dense1 = WeightSumBuild(_units[0])
        self.Dense2 = WeightSumBuild(_units[1])
        self.Dense_Out = WeightSumBuild(10)
    
    def call(self, Input):
        x = self.Dense1(Input)
        x = tf.nn.relu(x)
        x = self.Dense2(x)
        x = tf.nn.relu(x)
        return self.Dense_Out(x)

