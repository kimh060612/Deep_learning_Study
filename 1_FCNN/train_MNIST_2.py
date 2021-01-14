import tensorflow as tf
import numpy as np
from tensorflow import keras
from model.FCNN2 import FCNN

EPOCHS = 15
LR = 0.01
BatchSize = 32

# 이미지 불러오기
mnist = keras.datasets.mnist
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# 필요한 전처리
train_img, test_img = train_img.reshape([-1, 784]), test_img.reshape([-1, 784])
train_img = train_img.astype(np.float32) / 255.
test_img = test_img.astype(np.float32) / 255.

# Train-Validation Split
validation_img = train_img[-18000:]
validation_label = train_labels[-18000:]
train_img = train_img[:-18000]
train_labels = train_labels[:-18000]

# Train Data의 규합.
train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BatchSize)

# Validation Data의 규합
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_img, validation_label))
validation_dataset = validation_dataset.batch(BatchSize)

# Optimizer & Loss Function 정의
optimizer = keras.optimizers.Adam(learning_rate=LR)
loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_accuracy = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

model = FCNN()

for epoch in range(EPOCHS):
    print("Epoch %d start"%epoch)
    for step, (x_batch, y_batch) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_val = loss_function(y_batch, logits)
        grad = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))
        train_accuracy.update_state(y_batch, logits)
        if step % 500 == 0 :
            print("Training loss at step %d: %.4f"%(step, loss_val))

    train_acc = train_accuracy.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    train_accuracy.reset_states()

    for x_batch_val, y_batch_val in validation_dataset:
        val_logits = model(x_batch_val, training = False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    print("Validation acc: %.4f" % (float(val_acc),))
    val_acc_metric.reset_states()

