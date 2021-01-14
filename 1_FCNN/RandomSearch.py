from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

mnist = keras.datasets.mnist
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()
train_img, test_img = train_img.reshape([-1, 784]), test_img.reshape([-1, 784])
train_img = train_img.astype(np.float32) / 255.
test_img = test_img.astype(np.float32) / 255.

param_distribution = {
    "n_hidden": [0,1,2,3],
    "n_neurons": np.arange(1,100),
    "lr": reciprocal(3e-4, 3e-2)
}

def Build_model(n_hidden = 1, n_neurons=30, lr = 3e-3, input_shape=[784]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(units=10, activation="softmax"))
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    return model

keras_classify = keras.wrappers.scikit_learn.KerasClassifier(Build_model)
rnd_search_model = RandomizedSearchCV(keras_classify, param_distributions=param_distribution, n_iter = 10, cv = 3)
rnd_search_model.fit(train_img, train_labels, epochs=100, validation_data=(test_img,test_labels), callbacks=[keras.callbacks.EarlyStopping(patience=10)])

print(rnd_search_model.best_params_)
