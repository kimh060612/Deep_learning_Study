import tensorflow as tf
from tensorflow import keras
from .attention import Attention

class Encoder(keras.Model):
    def __init__(self, WORDS_NUM, emb_dim, hidden_unit, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.hidden_unit = hidden_unit
        self.embedding = keras.layers.Embedding(WORDS_NUM, emb_dim)
        self.LSTM = keras.layers.LSTM(hidden_unit, return_state=True, return_sequences=True)

    def call(self, inputs, enc_hidden = None):
        x = self.embedding(inputs)
        y, h, _ = self.LSTM(x, initial_state=enc_hidden)
        return y, h

class Decoder(keras.Model):
    def __init__(self, WORDS_NUM, emb_dim, hidden_unit, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = keras.layers.Embedding(WORDS_NUM, emb_dim)
        self.lstm = tf.keras.layers.LSTM(hidden_unit, return_sequences=True, return_state=True)
        self.attention = Attention()
        self.dense = tf.keras.layers.Dense(WORDS_NUM, activation='softmax')

    def call(self, x, hidden, mask=None):
        x = self.embedding(x)
        _, h, c = self.lstm(x)
        context_vec, attention_weight = self.attention(h, hidden, hidden, mask=mask)
        
        x_ = tf.concat([context_vec, c], axis=-1)
        out = self.dense(x_)
        return out, h, attention_weight
        