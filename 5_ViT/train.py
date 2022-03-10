import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from model.model import ViT
from utils.scheduler import CustomSchedule
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
# setting variables
IMAGE_SIZE=32
PATCH_SIZE=4 
NUM_LAYERS=8
NUM_HEADS=16
MLP_DIM=128
lr=1e-3
WEIGHT_DECAY=1e-4
BATCH_SIZE=64
D_MODEL=64
epochs=1

#Load the dataset
ds = tfds.load("cifar10", as_supervised=True)
ds_train = (
    ds["train"]
    .cache()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
ds_test = (
    ds["test"]
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

model = ViT(patch_size=PATCH_SIZE, 
            out_dim=10, 
            mlp_dim=MLP_DIM, 
            num_layer=NUM_LAYERS, 
            d_model=D_MODEL, 
            num_haed=NUM_HEADS, 
            d_ff=MLP_DIM, 
            drop_out_prob=0.1
)

learning_rate = CustomSchedule(D_MODEL)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
early_stop = keras.callbacks.EarlyStopping(patience=10)

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=epochs,
    callbacks=[early_stop],
)
if not os.path.exists("./ViT"):
    os.mkdir("./ViT")
model.save(os.path.join('./ViT', "ViT"))