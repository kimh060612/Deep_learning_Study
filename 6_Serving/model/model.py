from errno import ECOMM
import tensorflow as tf
from tensorflow import keras
from .layer import Encoder, GeneratePatch

class ViT(keras.Model):
    def __init__(self, patch_size, out_dim, mlp_dim, 
                 num_layer, d_model, num_haed, d_ff, drop_out_prob = 0.2):
        super(ViT, self).__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        
        self.patch_gen = GeneratePatch(patch_size=patch_size)
        self.linear_proj = keras.layers.Dense(d_model, activation="relu")
        
        self.encoders = [ Encoder(d_model=d_model, num_head=num_haed, d_ff=d_ff, drop_out_prob=drop_out_prob) for _ in range(num_layer) ]
        
        self.dense_mlp = keras.layers.Dense(mlp_dim, activation="relu")
        self.dropout = keras.layers.Dropout(drop_out_prob)
        self.dense_out = keras.layers.Dense(out_dim, activation="softmax")
    
    def build(self, x_shape):
        num_patch = (x_shape[1] * x_shape[2]) // (self.patch_size * self.patch_size) 
        
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patch + 1, self.d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.d_model)) 

    def call(self, x, training):
        
        patches = self.patch_gen(x)
        x = self.linear_proj(patches)
        
        batch_size = tf.shape(x)[0]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])    
        x = tf.concat([class_emb, x], axis=1)
        
        x = x + self.pos_emb
        
        for layer in self.encoders:
            x = layer(x, training)
        
        x = self.dense_mlp(x[:, 0, :])
        x = self.dropout(x)
        x = self.dense_out(x)
        
        return x