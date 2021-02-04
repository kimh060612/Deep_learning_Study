from loss import CustomMSE
from tensorflow import keras
import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def get_model(Input):
    X = keras.layers.Dense(10, activation='relu')(Input)
    X = keras.layers.Dense(5, activation='relu')(X)
    Out = keras.layers.Dense(1, activation='linear')(X)
    return Out

Opt = keras.optimizers.Adam(learning_rate=0.01)
Loss = CustomMSE()

x=[1,2,3,4,5,6,7,8,9,10]
x=np.asarray(x,dtype=np.float32).reshape((10,1))
y=[1,4,9,16,25,36,49,64,81,100]
y=np.asarray(y,dtype=np.float32).reshape((10,1))

inputs = keras.layers.Input(shape=(1, ))
output = get_model(inputs)

model = keras.models.Model(inputs = inputs, outputs = output)
model.compile(Opt, loss=Loss, metrics=["mse"])

model.fit(x, y, epochs=10)