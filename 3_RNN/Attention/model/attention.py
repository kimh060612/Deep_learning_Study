import tensorflow as tf
from tensorflow import keras

class Encoder(keras.Model):
    def __init__(self, WORDS_NUM, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = keras.layers.Embedding(WORDS_NUM, 64)
        self.LSTM = keras.layers.LSTM(512, return_state=True)

    def call(self, inputs, training=False, mask=None):
        x = self.embedding(inputs)
        _, h, c = self.LSTM(x)
        return h, c

class Decoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def call(self, inputs):
        pass

class Seq2SeqAttention(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def call(self, inputs):
        pass