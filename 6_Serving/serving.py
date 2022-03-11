import tensorflow as tf
from tensorflow import keras
from model.model import ViT
import flask
import requests

# setting variables
IMAGE_SIZE=32
PATCH_SIZE=4 
NUM_LAYERS=8
NUM_HEADS=16
MLP_DIM=128
lr=1e-3
BATCH_SIZE=64
D_MODEL=64

model = ViT(patch_size=PATCH_SIZE, 
            out_dim=10, 
            mlp_dim=MLP_DIM, 
            num_layer=NUM_LAYERS, 
            d_model=D_MODEL, 
            num_haed=NUM_HEADS, 
            d_ff=MLP_DIM, 
            drop_out_prob=0.1
)

