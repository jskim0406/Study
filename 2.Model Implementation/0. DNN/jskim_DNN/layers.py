from typing import List

import numpy as np
from numpy import ndarray

from .activations import Linear, Sigmoid, Tanh, ReLU
from .base import Operation, ParamOperation
from .dense import WeightMultiply, BiasAdd
from .dropout import Dropout
from jskim_DNN.utils.np_utils import assert_same_shape

class Layer(object):
    '''
    신경망 모델의 층 역할을 하는 클래스
    '''

    def __init__(self, neurons :int, dropout: float = 1.0):
        '''
        뉴런의 갯수 생성
        '''
        self.neurons = neurons
        self.first = True
        self.params :List[np.ndarray] = []
        self.param_grads :List[np.ndarray] = []
        self.operations :List[Operation] = []

    def _setup_layer(self, num_in: int)->None:
        '''
        Layer를 구현하는 concrete class는 _setup_layer 메서드를 구현해야 함
        '''
        raise NotImplementedError()

    def forward(self, input_ :np.ndarray, inference=False)->np.ndarray:
        '''
        입력값을 각 연산에 순서대로 통과시켜 순방향 계산을 수행함
        '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_
        # Operation.forward(): 안에서 self.input_ 이 update됨
        for operation in self.operations:
            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad :np.ndarray)->np.ndarray:
        '''
        output_grad를 각 연산에 역순으로 통과시켜 backprop 시행
        계산하기 전, 행렬의 모양을 확인한다.
        forward모두 수행 후(self.operations 모두 채워진 상태), backward 수행
        '''
        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self):
        '''
        각 operation 객체에서 _param_grad값을 꺼내 저장
        '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):
        '''
        각 operation 객체에서 _param값을 꺼내 저장
        '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

from typing import*

class Dense(Layer):
  '''
  Layer 클래스의 sequence(self.operations)의 concrete class
  Dense : Layer들의 fully_connected 구성
  '''

  def __init__(self, neurons :int, dropout: float = 1.0, activation = Sigmoid(),
               weight_init: str = "standard")->None:
      '''
      Dense 생성(초기화) 시, Layer의 상속 외에도
      activation을 추가로 지정해야 함
      '''
      super().__init__(neurons)
      self.activation = activation
      self.dropout = dropout
      self.weight_init = weight_init

  def _setup_layer(self, input_: ndarray) -> None:
      np.random.seed(self.seed)
      num_in = input_.shape[1]

      if self.weight_init == "glorot":
          scale = 2/(num_in + self.neurons)
      else:
          scale = 1.0

      # weights
      self.params = []
      self.params.append(np.random.normal(loc=0,
                                          scale=scale,
                                          size=(num_in, self.neurons)))

      # bias
      self.params.append(np.random.normal(loc=0,
                                          scale=scale,
                                          size=(1, self.neurons)))

      self.operations = [WeightMultiply(self.params[0]),
                         BiasAdd(self.params[1]),
                         self.activation]

      if self.dropout < 1.0:
          self.operations.append(Dropout(self.dropout))

      return None
