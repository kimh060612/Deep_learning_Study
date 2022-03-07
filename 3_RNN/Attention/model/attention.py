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
    def __init__(self, WORDS_NUM, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = keras.layers.Embedding(WORDS_NUM, 64)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(WORDS_NUM, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        x, h, c = inputs
        x = self.embedding(x)
        x, h, c = self.lstm(x, initial_state = [h, c])
        return self.dense(x), h, c

class Seq2Seq(keras.Model):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        pass