from numpy import ndarray
import numpy as np

from jskim_DNN.utils.np_utils import assert_same_shape

class Operation:
    '''
    신경망 모델의 연산의 최상위 Abstracdt class
    '''
    def __init__(self):
        pass

    def forward(self, input_: np.ndarray, inference: bool=False)->np.ndarray:
        self.input_ = input_
        self.output = self._output(inference)
        return self.output

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        '''
        self._input_grad() 호출함
        이때, 모양의 일치여부 확인 필요
        '''

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self, inference: bool)->np.ndarray:
        '''
        Operation클래스의 concrete class(Subclasses of ParamOperation class)에서
        _output 메서드 구현해야 함
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray)->np.ndarray:
        '''
        Operation을 구현한 모든 구상 클래스는 _input_grad 메서드를 구현해야 한다.
        '''
        raise NotImplementedError()


class ParamOperation(Operation):
    '''
    파라미터를 갖는 모든 연산의 Abstract class
    '''

    def __init__(self, param: np.ndarray):
        '''
        생성자 메서드
        '''
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        '''
        self._input_grad(), self._output_grad() 호출
        이때, 모양의 일치 여부 확인 필요
        '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self,output_grad):
        '''
        Operation을 구현한 모든 구상 클래스는 _param_grad 메서드를 구현해야 한다.
        '''
        raise NotImplementedError()
