import numpy as np
from numpy import ndarray
import numpy as np
from .base import Operation

from typing import List

class Sigmoid(Operation):
    '''
    sigmoid 활성화 함수
    '''

    def __init__(self)->None:
        '''pass'''
        super().__init__()

    def _output(self,inference: bool)->np.ndarray:
        '''
        출력값 계산
        '''
        return 1.0/(1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad :np.ndarray)->np.ndarray:
        '''
        입력에 대한 gradient 계산
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        # * 연산 순서 전환 디버깅함
        input_grad = sigmoid_backward*output_grad
        return input_grad

class Linear(Operation):
    '''
    항등 활성화 함수
    '''

    def __init__(self) -> None:
        '''기반 클래스의 생성자 메서드 실행'''
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        '''입력을 그대로 출력'''
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''그대로 출력'''
        return output_grad

class Tanh(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad * (1 - self.output * self.output)

class ReLU(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        mask = self.output >= 0
        return output_grad * mask
