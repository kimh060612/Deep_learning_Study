from model.layer import ResidualConvBlock, ResidualIdentityBlock
from tensorflow import keras
import tensorflow as tf

class ResNet50(keras.Model):
    def __init__(self):
        super().__init__()
        # Input Shape 224*224*3
        # Conv 1 Block
        self.ZeroPadding1 = keras.layers.ZeroPadding2D(padding=(3,3))
        self.Conv1 = keras.layers.Conv2D(filters = 64, kernel_size=(7, 7), strides=(2, 2))
        self.Batch1 = keras.layers.BatchNormalization()
        self.ReLU1 = keras.layers.LeakyReLU()
        self.ZeroPadding2 = keras.layers.ZeroPadding2D(padding=(1,1))

        self.MaxPool1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
        self.ResConvBlock1 = ResidualConvBlock(64, 256, strides = (1, 1))
        self.ResIdentityBlock1 = ResidualIdentityBlock(64, 256)
        self.ResIdentityBlock2 = ResidualIdentityBlock(64, 256)

        self.ResConvBlock2 = ResidualConvBlock(128, 512, strides = (2, 2))
        self.ResIdentityBlock3 = ResidualIdentityBlock(128, 512)
        self.ResIdentityBlock4 = ResidualIdentityBlock(128, 512)
        self.ResIdentityBlock5 = ResidualIdentityBlock(128, 512)

        self.ResConvBlock3 = ResidualConvBlock(256, 1024, strides = (2, 2))
        self.ResIdentityBlock6 = ResidualIdentityBlock(256, 1024)
        self.ResIdentityBlock7 = ResidualIdentityBlock(256, 1024)
        self.ResIdentityBlock8 = ResidualIdentityBlock(256, 1024)
        self.ResIdentityBlock9 = ResidualIdentityBlock(256, 1024)
        self.ResIdentityBlock10 = ResidualIdentityBlock(256, 1024)

        self.ResConvBlock4 = ResidualConvBlock(512, 2048, strides = (1, 1))
        self.ResIdentityBlock11 = ResidualIdentityBlock(512, 2048)
        self.ResIdentityBlock12 = ResidualIdentityBlock(512, 2048)
        
        self.GAP = keras.layers.GlobalAveragePooling2D()
        self.DenseOut = keras.layers.Dense(1000, activation='softmax')

    def call(self, Input):
        X = self.ZeroPadding1(Input)
        X = self.Conv1(X)
        X = self.Batch1(X)
        X = self.ReLU1(X)
        X = self.ZeroPadding2(X)

        X = self.MaxPool1(X)
        X = self.ResConvBlock1(X)
        X = self.ResIdentityBlock1(X)
        X = self.ResIdentityBlock2(X)

        X = self.ResConvBlock2(X)
        X = self.ResIdentityBlock3(X)
        X = self.ResIdentityBlock4(X)
        X = self.ResIdentityBlock5(X)

        X = self.ResConvBlock3(X)
        X = self.ResIdentityBlock6(X)
        X = self.ResIdentityBlock7(X)
        X = self.ResIdentityBlock8(X)
        X = self.ResIdentityBlock9(X)
        X = self.ResIdentityBlock10(X)

        X = self.ResConvBlock4(X)
        X = self.ResIdentityBlock11(X)
        X = self.ResIdentityBlock12(X)

        X = self.GAP(X)
        Out = self.DenseOut(X)
        
        return Out