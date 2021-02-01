from tensorflow import keras
import tensorflow as tf

#Layers without Aux output
class Inception(keras.layers.Layer):
    def __init__(self, Depth_list, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.Conv1x1_1 = keras.layers.Conv2D(Depth_list[0], kernel_size=(1, 1), activation='relu', padding="same")
        self.Conv1x1_2 = keras.layers.Conv2D(Depth_list[1], kernel_size=(1, 1), activation='relu', padding="same")
        self.Conv3x3_1 = keras.layers.Conv2D(Depth_list[2], kernel_size=(3, 3), padding="same", activation='relu')
        self.Conv1x1_3 = keras.layers.Conv2D(Depth_list[3], kernel_size=(1, 1), activation='relu', padding="same")
        self.Conv5x5_1 = keras.layers.Conv2D(Depth_list[4], kernel_size=(5, 5), padding="same", activation='relu')
        self.MaxPooling3x3_1 = keras.layers.MaxPool2D(pool_size=(3, 3), padding="same", strides=(1, 1))
        self.Conv1x1_4 = keras.layers.Conv2D(Depth_list[5], kernel_size=(1, 1), activation='relu', padding="same")

    def call(self, Input):
        X_1 = self.Conv1x1_1(Input)
        X_2 = self.Conv1x1_2(Input)
        X_2 = self.Conv3x3_1(X_2)
        X_3 = self.Conv1x1_3(Input)
        X_3 = self.Conv5x5_1(X_3)
        X_4 = self.MaxPooling3x3_1(Input)
        X_4 = self.Conv1x1_4(X_4)
        Output = keras.layers.Concatenate()([X_1, X_2, X_3, X_4])
        return Output

class InceptionWithAuxOut(keras.layers.Layer):
    def __init__(self, Depth_list, num_out, num_out_filter, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.Conv1x1_1 = keras.layers.Conv2D(Depth_list[0], kernel_size=(1, 1), activation='relu', padding="same")
        self.Conv1x1_2 = keras.layers.Conv2D(Depth_list[1], kernel_size=(1, 1), activation='relu', padding="same")
        self.Conv3x3_1 = keras.layers.Conv2D(Depth_list[2], kernel_size=(3, 3), padding="same", activation='relu')
        self.Conv1x1_3 = keras.layers.Conv2D(Depth_list[3], kernel_size=(1, 1), activation='relu', padding="same")
        self.Conv5x5_1 = keras.layers.Conv2D(Depth_list[4], kernel_size=(5, 5), padding="same", activation='relu')
        self.MaxPooling3x3_1 = keras.layers.MaxPool2D(pool_size=(3, 3), padding="same", strides=(1, 1))
        self.Conv1x1_4 = keras.layers.Conv2D(Depth_list[5], kernel_size=(1, 1), activation='relu', padding="same")

        self.Conv1x1_5 = keras.layers.Conv2D(num_out_filter, kernel_size=(1, 1), activation='relu')
        self.GAP = keras.layers.GlobalAveragePooling2D()
        self.Dense = keras.layers.Dense(num_out, activation='softmax')

    def call(self, Input):
        X_1 = self.Conv1x1_1(Input)
        X_2 = self.Conv1x1_2(Input)
        X_2 = self.Conv3x3_1(X_2)
        X_3 = self.Conv1x1_3(Input)
        X_3 = self.Conv5x5_1(X_3)
        X_4 = self.MaxPooling3x3_1(Input)
        X_4 = self.Conv1x1_4(X_4)
        Output = keras.layers.Concatenate()([X_1, X_2, X_3, X_4])
        
        X_Aux = self.Conv1x1_5(Input)
        X_Aux = self.GAP(X_Aux)
        Output_Aux = self.Dense(X_Aux)

        return Output, Output_Aux