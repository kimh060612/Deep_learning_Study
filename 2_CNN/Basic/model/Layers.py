from tensorflow import keras
import numpy as np
import tensorflow as tf

# 지난 시간에 했던 내용
# tensorflow 어떤 라이브러리인지 ==> 수치해석 라이브러리 딥러닝 
# 미분 h = 0.0000000001 f'(x) = (f(x + h) - f(x))/h
# gradient descent 미분을 결국 구해야함.
# 미분을 빠르게 구할 수 있는 AutoGrad 



class CustomConv2D(keras.layers.Layer):
    #              1. output image의 채널  2. 커널의 이미지 사이즈  3. Stride를 정해줬어야함. 4. Pooling을 정해줬어야함.(Optional) 5. Padding을 정해야함.
    #                                                       i  x 방향으로의 stride y 방향으로의 stride      i
    # "SAME" OH = H, "VALID" 가능한 패딩 필터 중에서 가장 작은 패딩(양수)으로 설정  
    # OH = (H + 2*P - KH)/S + 1 = 15.5
    def __init__(self, out_channel, kernel_size, Strides = (1, 1, 1, 1), Padding = "SAME", trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        # "3,4" 
        self.out_channel = out_channel

        if type(kernel_size) == type(1):
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) == type((1,1)):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Not a Valid Kernel Size")

        if type(Strides) == type(1):
            self.Stride = (1, Strides, Strides, 1)
        elif type(Strides) == type(tuple()):
            self.Stride = Strides
        else:
            raise ValueError("Not a Valid Kernel Size")
        
        if type(Padding) == type(str()):
            self.Padding = Padding
        else :
            raise ValueError("Not a Valid Kernel Size")

    def build(self, input_shape):
        WeightShape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.out_channel)
        self.Kernel = self.add_weight(
            shape=WeightShape,
            initializer="random_normal",
            trainable= True
        )

        self.Bias = self.add_weight(
            shape=(self.out_channel, ),
            initializer="random_normal",
            trainable=True
        )
        

    def call(self, Input):
        Out = tf.nn.conv2d(Input, self.Kernel, strides=self.Stride, padding=self.Padding)
        Out = tf.nn.bias_add(Out, self.Bias, data_format="NHWC")
        return Out


# Layer class를 상속을 받을때 주로 오버라이딩을 받아서 사용해야하는 함수
# 1. 생성자
#   얘는 그냥 정의.
# 2. build
#   call이 실행되기 전에 무조건 호출되는 함수
# 3. call
#   Model의 Feed Forward (신경망에 입력을 넣어서 출력을 뽑아내는 과정)를 담당하는 함수
# 이미지 => 딥러닝 층층이 쌓여져 있음 => 출력 앞으로 먹임. Feed Forward

class WeightSumBuild(keras.layers.Layer):
    def __init__(self, _units = 32, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.units = _units
    
    def build(self, input_dim):
        # (60000, 784)
        self.Weight = self.add_weight(
            shape=(input_dim[-1], self.units),
            initializer = "random_normal",
            trainable= True
        )

        self.Bias = self.add_weight(
            shape = (self.units,),
            initializer = "random_normal",
            trainable = True
        )
    
    def call(self, Input):
        return tf.matmul(Input, self.Weight) + self.Bias