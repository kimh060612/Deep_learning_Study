import tensorflow as tf
import numpy as np
from tensorflow import keras
from Optimizer import CustomSGDOptimizer

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

# Functional API : 가장 흔하게 볼 수 있는 tensorflow 모델의 정의 방식
def DeepModel(Input):
    X = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(Input)
    X = keras.layers.MaxPool2D(pool_size=(2,2), padding="SAME")(X)
    X = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(2,2), padding="SAME")(X)
    X = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(128, activation='relu')(X)
    X = keras.layers.Dense(10, activation='softmax')(X)
    return X

EPOCHS = 15
LR = 0.001
BatchSize = 32

# 이미지 불러오기
mnist = keras.datasets.mnist
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# 필요한 전처리
train_img, test_img = train_img.reshape([-1, 28, 28, 1]), test_img.reshape([-1, 28, 28, 1])
train_img = train_img.astype(np.float32) / 255.
test_img = test_img.astype(np.float32) / 255.

# Train-Validation Split
validation_img = train_img[-18000:]
validation_label = train_labels[-18000:]
train_img = train_img[:-18000]
train_labels = train_labels[:-18000]

# Train Data의 규합.
                                                    # Training Image랑 Training Label이랑 묶음.
train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_labels))
                            # 섞어.                        # 배치 사이즈 만큼 나눠.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BatchSize)
# 60000, 28, 28, 1
# 1875, 32, 28, 28, 1
# 1875, (N, H, W, C)

# Validation Data의 규합
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_img, validation_label))
validation_dataset = validation_dataset.batch(BatchSize)

# Optimizer & Loss Function 정의
optimizer = keras.optimizers.Adam(learning_rate=LR)
#optimizer = CustomSGDOptimizer(lr=LR)
loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Sparse의 뜻: 희소한 ==> Sparse Matrix: 원소의 대부분이 0인 행렬
# Sparse Categorical Cross-Entropy vs Categorical Cross-Entropy
#           1                            [0, 1, 0, 0, 0, ... , 0]
#       (128, 1)                             (128, 10)

train_accuracy = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

inputs = keras.layers.Input(shape=(28, 28, 1))
outputs = DeepModel(inputs)
model = keras.models.Model(inputs = inputs, outputs = outputs)


# Custom training Loop
# 앞으로 볼 대부분의 논문의 오픈 소스들이 이렇게 training loop를 정의함.
# keras에서 사용할 수 있는 fit은 너무 제한적임.
# fit의 대표적인 한계점: Optimzier 2개이상 못씀.
for epoch in range(EPOCHS):
    print("Epoch %d start"%epoch)
    # step, 1개의 batch ==> 의사 코드에서 batch 뽑는 역할
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        # **********************************************************************************************
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_val = loss_function(y_batch, logits)
            # 여기서 신경망의 Feed Forward & Expectation of Loss 계산을 진행함.
        # tape.gradient를 호출하면 ==> gradient 계산이 진행됨.
        grad = tape.gradient(loss_val, model.trainable_weights)
        # 정의된 Optimizer를 이용해서 Update를 진행함.
        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        # train에서의 정확도를 계산함.
        train_accuracy.update_state(y_batch, logits)
        if step % 500 == 0 :
            print("Training loss at step %d: %.4f"%(step, loss_val))
    
    # 정확도 뽑아 보겠다.
    train_acc = train_accuracy.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    train_accuracy.reset_states()

    for x_batch_val, y_batch_val in validation_dataset:
        val_logits = model(x_batch_val, training = False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    print("Validation acc: %.4f" % (float(val_acc),))
    val_acc_metric.reset_states()

