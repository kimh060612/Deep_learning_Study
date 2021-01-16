import tensorflow as tf
from tensorflow import keras

class WeightSum(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        # tf.Variable을 활용하는 예시
        w_init = tf.random_normal_initializer()
        # 1. 학습 시킬 행렬을 직접 정의함.
        self.Weight = tf.Variable(
            initial_value = w_init(shape=(input_dim, units), dtype="float32"),
            trainable = True
        )
        b_init = tf.zeros_initializer()
        self.Bias = tf.Variable(
            initial_value = b_init(shape=(units, ), dtype="float32"),
            trainable = True
        )
        # add_weight를 활용하는 예시 - Short Cut
        '''
        self.Weight = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.Bias = self.add_weight(shape=(units,), initializer="zeros", trainable=True)
        '''
    
    def call(self, Input):
        return tf.matmul(Input, self.Weight) + self.Bias

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

# 784
# 1024*540 480*240 1920*1080 ....... ==> (416 * 416)