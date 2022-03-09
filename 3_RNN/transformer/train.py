import tensorflow as tf
from tensorflow import keras
from model.model import Transformer
from utils.scheduler import CustomSchedule
from sklearn.model_selection import train_test_split
import json
import unicodedata
import re
import os
import io
import time

EPOCHS = 20
BUFFER_SIZE = 20000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
tokenizers = None
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

if __name__ == "__main__":
    
    config_file = open("./config.json", "r").read()
    config_ = json.loads(config_file)
    
    path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)
    path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

    num_examples = 30000
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
    max_sequence_length = max(max_length_targ, max_length_inp)
    
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1
    
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    learning_rate = CustomSchedule(config_["d_model"])
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.Mean(name='train_accuracy')
    transformer = Transformer(
            num_layers=config_["num_layer"],
            d_model=config_["d_model"],
            num_head=config_["num_head"],
            d_ff=config_["d_ff"],
            input_voc_size=vocab_inp_size,
            output_voc_size=vocab_tar_size,
            max_seq_len_in=1000,
            max_seq_len_out=1000,
            drop_out_prob=config_["drop_out"])
    
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            
            inp = keras.preprocessing.sequence.pad_sequences(inp, maxlen=max_sequence_length, padding="post", truncating="post")
            tar_inp = keras.preprocessing.sequence.pad_sequences(tar_inp, maxlen=max_sequence_length, padding="post", truncating="post")
            tar_real = keras.preprocessing.sequence.pad_sequences(tar_real, maxlen=max_sequence_length, padding="post", truncating="post")
            
            with tf.GradientTape() as tape:
                predictions = transformer([inp, tar_inp],
                                            training = True)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')