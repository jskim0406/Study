import numpy as np
from numpy import ndarray

from jskim_DNN.utils.np_utils import (assert_same_shape,
                       softmax,
                       normalize,
                       #exp_ratios,
                       unnormalize)

class Loss(object):
   '''
   신경망 모델의 손실을 계산하는 클래스
   '''

   def __init__(self):
       '''pass'''
       pass

   def forward(self, prediction :np.ndarray, target :np.ndarray)->float:
       '''
       실제 Loss값 계산
       '''
       # 새로 생성#

       assert_same_shape(prediction,target)

       self.prediction = prediction
       self.target = target

       loss_value = self._output()

       return loss_value

   def backward(self)->np.ndarray:
       '''
       손실함수의 입력값에 대해 손실의 gradient 계산함
       '''
       self.input_grad = self._input_grad()

       assert_same_shape(self.prediction, self.input_grad)

       return self.input_grad

   def _output(self)->float:
       '''
       Loss class를 확장한 concrete class는 _output 메서드를 구현해야 함
       '''
       raise NotImplementedError

   def _input_grad(self)->np.ndarray:
       '''
       Loss class를 확장한 concrete class는 _input_grad 메서드를 구현해야 함
       '''
       raise NotImplementedError


class MeanSquaredError(Loss):

   def __init__(self):
       super().__init__()

   def _output(self)->float:
       loss = (np.sum(np.power(self.prediction - self.target,2))/self.prediction.shape[0])
       return loss

   def _input_grad(self)->np.ndarray:
       return 2.0*(self.prediction - self.target) / self.prediction.shape[0]

class SoftmaxCrossEntropy(Loss):
   def __init__(self, eps: float=1e-9) -> None:
       super().__init__()
       self.eps = eps
       self.single_class = False

   def _output(self) -> float:

       # if the network is just outputting probabilities
       # of just belonging to one class:
       if self.target.shape[1] == 0:
           self.single_class = True

       # if "single_class", apply the "normalize" operation defined above:
       if self.single_class:
           self.prediction, self.target = \
           normalize(self.prediction), normalize(self.target)

       # applying the softmax function to each row (observation)
       softmax_preds = softmax(self.prediction, axis=1)

       # clipping the softmax output to prevent numeric instability
       self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

       # actual loss computation
       softmax_cross_entropy_loss = (
           -1.0 * self.target * np.log(self.softmax_preds) - \
               (1.0 - self.target) * np.log(1 - self.softmax_preds)
       )

       return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

   def _input_grad(self) -> ndarray:

       # if "single_class", "un-normalize" probabilities before returning gradient:
       if self.single_class:
           return unnormalize(self.softmax_preds - self.target)
       else:
           return (self.softmax_preds - self.target) / self.prediction.shape[0]
