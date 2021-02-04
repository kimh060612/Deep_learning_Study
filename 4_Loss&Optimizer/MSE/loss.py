from tensorflow import keras
import tensorflow as tf

class CustomMSE(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        L = tf.math.square(y_true - y_pred)
        L = tf.math.reduce_mean(L)
        return L

class CustomRegularizer(keras.regularizers.Regularizer):
    def __init__(self, _factor):
        super().__init__()
        self.factor = _factor
    
    def call(self, weights):
        return tf.math.reduce_sum(tf.math.abs(self.factor * weights))
    
    def get_config(self):
        return {"factor" : self.factor}