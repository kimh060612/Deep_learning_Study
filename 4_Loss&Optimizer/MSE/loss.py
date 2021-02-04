from tensorflow import keras
import tensorflow as tf

class CustomMSE(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        L = tf.math.square(y_true - y_pred)
        L = tf.math.reduce_mean(L)
        return L
