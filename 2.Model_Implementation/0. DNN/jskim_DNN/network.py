from typing import List
from numpy import ndarray
import numpy as np
from .layers import Layer
from .losses import Loss, MeanSquaredError


class NeuralNetwork:
    '''
    신경망을 나타내는 클래스
    '''
    def __init__(self, layers :List[Layer], loss :Loss, seed :int = 1):
        '''
        신경망의 층, 손실함수를 정의
        '''
        self.layers = layers
        self.loss = loss
        self.seed = seed

        # layer가 갖고 있는 seed값을 self.seed로 변경하도록 하는 코드
        # setattr(object, name, value) := object에 존재하는 속성의 값을 바꾸거나, 새로운 속성을 생성하여 값을 부여함
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch :np.ndarray, inference: bool = False)->np.ndarray:
        '''
        데이터를 각 층에 순서대로 통과시킴(순방향 계산)
        '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out,inference)

        return x_out

    def backward(self, loss_grad :np.ndarray)->None:
        '''
        데이터를 각 층에 역순으로 통과시킴(역방향 계산)
        '''
        grad = loss_grad
        # reversed 하지않아 디버깅함
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self, x_batch :np.ndarray, y_batch :np.ndarray)->float:
        '''
        1. foward
        2. get_loss
        3. backward
        '''
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())

        return loss

    def params(self):
        '''
        신경망의 param 값을 받음
        '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
        신경망의 각 param에 대한 param_grad 값을 받음
        '''
        for layer in self.layers:
            yield from layer.param_grads
