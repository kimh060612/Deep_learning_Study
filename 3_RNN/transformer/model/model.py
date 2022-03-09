import tensorflow as tf
import numpy as np
from tensorflow import keras
from .layers import positional_encoding, Encoder, Decoder

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

class EncoderModel(keras.Model):
    def __init__(self, input_voc_size, num_layers, max_seq_len, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_voc_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, self.d_model)
        self.enc_layers = [ Encoder(d_model, num_head, d_ff, drop_out_prob) for _ in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(drop_out_prob)
        
    def call(self, x, training, mask):
        
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        return x
        
class DecoderModel(keras.Model):
    def __init__(self, output_voc_size, num_layers, max_seq_len, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super(DecoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(output_voc_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, self.d_model)
        
        self.dec_layers = [ Decoder(d_model, num_head, d_ff, drop_out_prob) for _ in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(drop_out_prob)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
        return x
    
class Transformer(keras.Model):
    def __init__(self, input_voc_size, output_voc_size, num_layers, max_seq_len_in, max_seq_len_out, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super().__init__()
        self.Encoder = EncoderModel(input_voc_size, num_layers, max_seq_len_in, d_model, num_head, d_ff, drop_out_prob)
        self.Decoder = DecoderModel(output_voc_size, num_layers, max_seq_len_out, d_model, num_head, d_ff, drop_out_prob)
        
        self.final_layer = tf.keras.layers.Dense(output_voc_size)
        
    def call(self, inputs, training):
        inp, tar = inputs
        
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.Encoder(inp, training, enc_padding_mask)
        dec_output = self.Decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        return final_output
        
    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask