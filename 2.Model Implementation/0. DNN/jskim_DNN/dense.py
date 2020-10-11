import numpy as np
from numpy import ndarray

from .base import ParamOperation


class WeightMultiply(ParamOperation):
    '''
    가중치 행렬곱 연산
    '''

    def __init__(self, W: np.ndarray):
        '''
        self.paran = W 로 초기화
        '''
        super().__init__(W)

    def _output(self,inference: bool)->np.ndarray:
        '''
        출력값 계산
        '''
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad :np.ndarray)->np.ndarray:
        '''
        입력에 대한 gradient 계산
        '''
        return np.dot(output_grad, np.transpose(self.param,(1,0)))

    def _param_grad(self, output_grad :np.ndarray)->np.ndarray:
        '''
        파라미터(W)에 대한 gradient 계산
        '''
        return np.dot(np.transpose(self.input_,(1,0)), output_grad)


class BiasAdd(ParamOperation):
    '''
    편향을 더하는 연산
    '''

    def __init__(self, B: np.ndarray):
        '''
        self.paran = B 로 초기화
        초기화 전에 행렬의 모양 확인 필요
        '''
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self,inference: bool)->np.ndarray:
        '''
        출력값 계산
        '''
        return self.input_ + self.param

    def _input_grad(self, output_grad :np.ndarray)->np.ndarray:
        '''
        입력값에 대한 gradient 계산
        '''
        # * 연산 순서 전환 디버깅함
        return np.ones_like(self.input_)*output_grad

    def _param_grad(self, output_grad :np.ndarray)->np.ndarray:
        '''
        파라미터에 대한 gradient 계산
        '''
        # * 연산 순서 전환 디버깅함
        param_grad = np.ones_like(self.param)*output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
