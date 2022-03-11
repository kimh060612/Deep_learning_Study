import tensorflow as tf
from tensorflow import keras
import numpy as np

class GeneratePatch(keras.layers.Layer):
    def __init__(self, patch_size):
        super(GeneratePatch, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images, 
                                        sizes=[1, self.patch_size, self.patch_size, 1], 
                                        strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #here shape is (batch_size, num_patches, patch_h*patch_w*c) 
        return patches

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_head, name="mha"):
        super(MultiHeadAttention, self).__init__(name=name)        
        if not d_model % num_head  == 0:
            raise ValueError("Invalid head and d_model!")
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.d_model = d_model
        
        self.Wq = keras.layers.Dense(self.d_model)
        self.Wk = keras.layers.Dense(self.d_model)
        self.Wv = keras.layers.Dense(self.d_model)
        
        self.dense = keras.layers.Dense(self.d_model, activation="relu")

    def split_head(self, batch_size, x):
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask = None):
        batch_size = tf.shape(q)[0]
        
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        # q: (batch_size, seq_len, d_model)
        # k: (batch_size, seq_len, d_model)
        # v: (batch_size, seq_len, d_model)
        
        q, k, v = self.split_head(batch_size, q), self.split_head(batch_size, k), self.split_head(batch_size, v)
        # q: (batch_size, num_head, seq_len, d_k)
        # k: (batch_size, num_head, seq_len, d_k)
        # v: (batch_size, num_head, seq_len, d_v)
        
        qkT = tf.matmul(q, k, transpose_b=True) # (batch_size, num_head, seq_len, seq_len)
        d_k = tf.cast(self.d_k, dtype=tf.float32)
        scaled_qkT = qkT / tf.math.sqrt(d_k)
        if not mask == None:
            scaled_qkT += (mask * -1e9)
        
        attention_dist = tf.nn.softmax(scaled_qkT, axis=-1)
        attention = tf.matmul(attention_dist, v) # (batch_size, num_head, seq_len, d_k)
        
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len, num_head, d_k)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.d_model)) # (batch_size, seq_len, d_model)
        output = self.dense(concat_attention)
        
        return output

class Encoder(keras.layers.Layer):
    def __init__(self, d_model, num_head, d_ff, drop_out_prob = 0.2, name="Encoder"):
        super(Encoder, self).__init__(name=name)
        self.d_model = d_model
        self.num_head = num_head
        num_model = name.split('_')[1]
        self.MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_head=num_head, name="mha_" + num_model)
        
        self.dense_1 = keras.layers.Dense(d_ff, activation="relu")
        self.dense_2 = keras.layers.Dense(d_model, activation="relu")
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(drop_out_prob)
        self.dropout2 = keras.layers.Dropout(drop_out_prob)

    def call(self, x, training=True, mask=None):
        
        out_atten = self.MultiHeadAttention(x, x, x, mask)
        out_atten = self.dropout1(out_atten, training=training)
        x = self.layernorm1(out_atten + x)
        
        out_dense = self.dense_1(x)
        out_dense = self.dense_2(out_dense)
        out_dense = self.dropout2(out_dense, training=training)
        x = self.layernorm2(out_dense + x)
        
        return x
    
