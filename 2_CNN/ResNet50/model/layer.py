from tensorflow import keras as tfk
import tensorflow as tf

class ResidualConvBlock(tfk.layers.Layer):
    def __init__(self, InputChannel, OutputChannel, strides = (1, 1), trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

        self.Batch1 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv1 = tfk.layers.Conv2D(filters=InputChannel, kernel_size=(1, 1), strides=strides)
        self.LeakyReLU1 = tfk.layers.LeakyReLU()
        self.Batch2 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv2 = tfk.layers.Conv2D(filters=InputChannel, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.LeakyReLU2 = tfk.layers.LeakyReLU()
        self.Batch3 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv3 = tfk.layers.Conv2D(filters=OutputChannel, kernel_size=(1, 1), strides=(1, 1))
        self.LeakyReLU3 = tfk.layers.LeakyReLU()

        # Skip Connection
        self.SkipConnection = tfk.layers.Conv2D(filters=OutputChannel, kernel_size=(1, 1), strides=strides)
        self.SkipBatch = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.LeakyReLUSkip = tfk.layers.LeakyReLU()

    def call(self, Input):
        Skip = Input
        Skip = self.SkipConnection(Skip)
        Skip = self.SkipBatch(Skip)
        Skip = self.LeakyReLUSkip(Skip)
        Z = Input
        Z = self.conv1(Z)
        Z = self.Batch1(Z)
        Z = self.LeakyReLU1(Z)
        Z = self.conv2(Z)
        Z = self.Batch2(Z)
        Z = self.LeakyReLU2(Z)  
        Z = self.conv3(Z)
        Z = self.Batch3(Z)
        Z = self.LeakyReLU3(Z)
        return Z + Skip 


class ResidualIdentityBlock(tfk.layers.Layer):
    def __init__(self, InputChannel, OutputChannel, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.Batch1 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv1 = tfk.layers.Conv2D(filters=InputChannel, kernel_size=(1, 1), strides=(1, 1))
        self.LeakyReLU1 = tfk.layers.LeakyReLU()
        self.Batch2 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv2 = tfk.layers.Conv2D(filters=InputChannel, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.LeakyReLU2 = tfk.layers.LeakyReLU()
        self.Batch3 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv3 = tfk.layers.Conv2D(filters=OutputChannel, kernel_size=(1, 1), strides=(1, 1))
        self.LeakyReLU3 = tfk.layers.LeakyReLU()
    
    def call(self, Input):
        Skip = Input
        Z = Input
        Z = self.conv1(Z)
        Z = self.Batch1(Z)
        Z = self.LeakyReLU1(Z)
        Z = self.conv2(Z)
        Z = self.Batch2(Z)
        Z = self.LeakyReLU2(Z)  
        Z = self.conv3(Z)
        Z = self.Batch3(Z)
        Z = self.LeakyReLU3(Z)
        # Z : 256
        return Z + Skip
