from tensorflow import keras
import numpy as np

class FCNN(keras.Model):
    def __init__(self, _units = [56, 56], _activation = ['relu', 'relu'], **kwargs):
        super().__init__(**kwargs)
        # 케라스에게 우리가 FCNN Layer를 만들겠다고 정의를 해주는 객체를 만드는 class.
        # kernel_initializer: Weight를 어떻게 초기화할 것인가를 결정하는 파라미터.
        self.Hidden1 = keras.layers.Dense(units=_units[0], activation=_activation[0], kernel_initializer="normal")
        self.Hidden2 = keras.layers.Dense(units=_units[1], activation=_activation[1], kernel_initializer="normal")
        self._output = keras.layers.Dense(units=10, activation="softmax")

    def call(self, Input):
        hidden1 = self.Hidden1(Input)
        hidden2 = self.Hidden2(hidden1)
        Output = self._output(hidden2)
        return Output


'''
오버 라이딩: 부모 클래스에 선언 or 정의되어 있는 메소드를 자식 클래스에서 재 정의를 하여 사용하는 방법

1. __init__
    다용도 -> 신경망에 필요한 정보들을 정의해주는
2. call
    신경에 데이터를 넣어서 결과를 얻어내는 함수 ==> Feed Forward를 하는 함수 
3. build
    call을 호출하기 전에 반드시 호출되는 함수.
'''