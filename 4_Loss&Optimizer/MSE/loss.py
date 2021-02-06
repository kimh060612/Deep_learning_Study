from tensorflow import keras
import tensorflow as tf

# Custom Loss
class CustomMSE(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        # tf.math.square : 성분별 제곱
        # [1,2,3] ==> [1,4,9]
        L = tf.math.square(y_true - y_pred)
        # tf.math.reduce_sum: 벡터 합.
        # [1,2,3] ==> 6
        L = tf.math.reduce_mean(L)
        return L

# Custom Regularizer(규제 ==> MAP)
class CustomRegularizer(keras.regularizers.Regularizer):
    def __init__(self, _factor):
        super().__init__()
        self.factor = _factor
    
    def call(self, weights):
        return tf.math.reduce_sum(tf.math.abs(self.factor * weights))
    
    def get_config(self):
        return {"factor" : self.factor} # 모델을 저장할때 custom layer, loss, 등등을 저장할때 같이 저장해 주는 역할.

# model.h5 (X)
# tensorflow 딴에서 저장해야함. .pb (O)