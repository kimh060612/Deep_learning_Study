import tensorflow as tf
import numpy as np
from tensorflow import keras
from model.model import RNNLayer

data_url = "https://homl.info/shakespeare"
file_path = keras.utils.get_file("shakespeare.txt", data_url)

with open(file_path, "r") as file:
    text_file = file.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(text_file)

max_id = len(tokenizer.word_index)
data_size = tokenizer.document_count

[encoded] = np.array(tokenizer.texts_to_sequences([text_file])) - 1

train_size = data_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_len = n_steps + 1
dataset = dataset.window(window_len, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_len))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)

model = RNNLayer(128, max_id)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(dataset, epochs=20, verbose=True)