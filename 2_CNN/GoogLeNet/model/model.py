from model.layer import Inception, InceptionWithAuxOut
from tensorflow import keras
import tensorflow as tf

inception3a_list = [64, 96, 128, 16, 32, 32]
inception3b_list = [128, 128, 192, 32, 96, 64]
inception4a_list = [192, 96, 208, 16, 48, 64]
inception4b_list = [160, 112, 224, 24, 64, 64]
inception4c_list = [128, 128, 256, 24, 64, 64]
inception4d_list = [112, 144, 288, 32, 64, 64]
inception4e_list = [256, 160, 320, 32, 128, 128]
inception5a_list = [256, 160, 320, 32, 128, 128]
inception5b_list = [384, 192, 394, 48, 128, 128]

class GooLeNetModel(keras.Model):
    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ZeroPad1 = keras.layers.ZeroPadding2D(padding=(3, 3))
        self.Conv1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
        self.ZeroPad1 = keras.layers.ZeroPadding2D(padding=(1, 1))
        self.Conv1x1_1 = keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu')
        self.Conv2 = keras.layers.Conv2D(192, kernel_size=(3, 3), padding="same", activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        self.inception3a = Inception(Depth_list = inception3a_list)
        self.inception3b = Inception(Depth_list = inception3b_list)
        self.maxpool3 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        self.inception4a = InceptionWithAuxOut(Depth_list = inception4a_list, num_out = n_classes, num_out_filter = 512) #
        self.inception4b = Inception(Depth_list = inception4b_list)
        self.inception4c = Inception(Depth_list = inception4c_list)
        self.inception4d = InceptionWithAuxOut(Depth_list = inception4d_list, num_out = n_classes, num_out_filter = 528) #
        self.inception4e = Inception(Depth_list = inception4e_list)
        self.maxpool4 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        self.inception5a = Inception(Depth_list = inception5a_list)
        self.inception5b = Inception(Depth_list = inception5b_list)

        self.GAP = keras.layers.GlobalAveragePooling2D()
        # self.dropout = keras.layers.dropout(0.4)
        self.Dense1 = keras.layers.Dense(1000, activation='linear')
        self.DenseOut = keras.layers.Dense(n_classes, activation='softmax')

    def call(self, Input):
        
        X = self.ZeroPad1(Input)
        X = self.Conv1(X)
        X = self.maxpool1(X)
        X = self.ZeroPad1(X)
        X = self.Conv1x1_1(X)
        X = self.Conv2(X)
        X = self.maxpool2(X)

        X = self.inception3a(X)
        X = self.inception3b(X)
        X = self.maxpool3(X)
        X, Out_Aux_1 = self.inception4a(X)
        X = self.inception4b(X)
        X = self.inception4c(X)
        X, Out_Aux_2 = self.inception4d(X)
        X = self.inception4e(X)
        X = self.maxpool4(X)
        X = self.inception5a(X)
        X = self.inception5b(X)

        X = self.GAP(X)
        X = self.Dense1(X)
        Out = self.DenseOut(X)

        return Out, Out_Aux_2, Out_Aux_1