import tensorflow as tf
import numpy as np
from tensorflow import keras

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model_save = d_model
        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        config = {
            'd_model': self.d_model_save,
            'warmup_steps': self.warmup_steps,
        }
        return config