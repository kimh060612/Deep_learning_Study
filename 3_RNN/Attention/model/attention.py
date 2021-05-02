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

class Seq2SeqAttention(keras.Model):
    def __init__(self, sos, eos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(WORDS_NUM=200)
        self.decoder = Decoder(WORDS_NUM=200)
        self.sos = sos
        self.eos = eos

    def call(self, inputs, training=False, mask=None):
        if training:
            x, y = inputs
            h, c = self.encoder(x)
            y, _, _ = self.decoder((y, h, c))
            return y
        else :
            x = inputs
            h, c = self.encoder(x)
            y = tf.convert_to_tensor(self.sos)
            y = tf.reshape(y, (1, 1))
            seq = tf.TensorArray(tf.int32, 64)
            for idx in tf.range(64):
                y, h, c = self.decoder([y, h, c])
                y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
                y = tf.reshape(y, (1, 1))
                seq = seq.write(idx, y)
                if y == self.eos:
                    break
            return tf.reshape(seq.stack(), (1, 64))
            