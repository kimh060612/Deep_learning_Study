import tensorflow as tf
import numpy as np
from tensorflow import keras

data_url = "https://homl.info/shakespeare"
file_path = keras.utils.get_file("shakespeare.txt", data_url)

with open(file_path, "r") as file:
    text_file = file.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_text(text_file)

max_id = len(tokenizer.word_index)
data_size = tokenizer.document_size

[encoded] = np.array(tokenizer.texts_to_sequences([text_file])) - 1


    