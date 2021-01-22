from tensorflow import keras
import numpy as np
import tensorflow as tf

class CustomConv2D(keras.layers.Layer):
    def __init__(self, out_channel, kernel_size, Strides = (1, 1, 1, 1), Padding = "SAME", trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        
        self.out_channel = out_channel

        if type(kernel_size) == type(1):
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) == type((1,1)):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Not a Valid Kernel Size")

        if type(Strides) == type(1):
            self.Stride = (1, Strides, Strides, 1)
        elif type(Strides) == type(tuple()):
            self.Stride = Strides
        else:
            raise ValueError("Not a Valid Kernel Size")
        
        if type(Padding) == type(str()):
            self.Padding = Padding
        else :
            raise ValueError("Not a Valid Kernel Size")
        

    def build(self, input_shape):
        
        WeightShape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.out_channel)
        self.Kernel = self.add_weight(
            shape=WeightShape,
            initializer="random_normal",
            trainable= True
        )

        self.Bias = self.add_weight(
            shape=(self.out_channel, ),
            initializer="random_normal",
            trainable=True
        )
        

    def call(self, Input):
        Out = tf.nn.conv2d(Input, self.Kernel, strides=self.Stride, padding=self.Padding)
        Out = tf.nn.bias_add(Out, self.Bias, data_format="NHWC")
        return Out


class WeightSumBuild(keras.layers.Layer):
    def __init__(self, _units = 32, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.units = _units
    
    def build(self, input_dim):
        # (60000, 784)
        self.Weight = self.add_weight(
            shape=(input_dim[-1], self.units),
            initializer = "random_normal",
            trainable= True
        )

        self.Bias = self.add_weight(
            shape = (self.units,),
            initializer = "random_normal",
            trainable = True
        )
    
    def call(self, Input):
        return tf.matmul(Input, self.Weight) + self.Bias