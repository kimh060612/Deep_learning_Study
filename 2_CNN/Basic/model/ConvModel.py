from model.Layers import CustomConv2D
from model.Layers import WeightSumBuild
from model.CONFIG import weights
from tensorflow import keras
import tensorflow as tf

class ConvModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.Conv1 = CustomConv2D(out_channel=weights['wc1'][-1], kernel_size=weights['wc1'][0])
        self.Conv2 = CustomConv2D(out_channel=weights['wc2'][-1], kernel_size=weights['wc2'][0])
        self.Conv3 = CustomConv2D(out_channel=weights['wc3'][-1], kernel_size=weights['wc3'][0])
        self.Dense1 = WeightSumBuild(_units=weights['wd1'])
        self.DenseOut = WeightSumBuild(_units=weights['out'])
    
    def call(self, Input):
        X = self.Conv1(Input)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool2d(X, ksize=[1,2,2,1], strides=(1,1,1,1), padding="SAME")
        X = self.Conv2(X)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool2d(X, ksize=[1,2,2,1], strides=(1,1,1,1), padding="VALID")
        X = self.Conv3(X)
        X = tf.nn.relu(X)
        X = keras.layers.Flatten()(X)
        X = self.Dense1(X)
        X = tf.nn.relu(X)
        X = self.DenseOut(X)
        Out_ = tf.nn.softmax(X)

        return Out_
        

