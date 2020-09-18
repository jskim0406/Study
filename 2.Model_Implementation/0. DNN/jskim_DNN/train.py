from typing import Tuple
from copy import deepcopy

import numpy as np
from numpy import ndarray

from .network import NeuralNetwork
from .optimizers import Optimizer
from jskim_DNN.utils.np_utils import permute_data

from copy import deepcopy
from typing import Tuple

class Trainer(object):
    '''
    신경망 모델을 학습시키는 역할을 수행함
    '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer) -> None:
        '''
        학습을 수행하려면 NeuralNetwork, Optimizer 객체가 필요함
        Optimizer 객체의 인스턴스 변수로 NeuralNetwork 객체를 전달할 것
        '''
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def generate_batches(self,
                         X: ndarray,
                         y: ndarray,
                         size: int = 32) -> Tuple[ndarray]:
        '''
        배치 생성
        '''
        assert X.shape[0] == y.shape[0], \
        '''
        특징과 목푯값은 행의 수가 같아야 하는데,
        특징은 {0}행, 목푯값은 {1}행이다
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch


    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            early_stopping: bool = True,
            restart: bool = True)-> None:
        '''
        일정 횟수의 에폭을 수행하며 학습 데이터에 신경망을 최적화함
        eval_every 변수에 설정된 횟수의 매 에폭마다 테스트 데이터로
        신경망의 예측 성능을 측정함
        '''

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e+1) % eval_every == 0:

                # 조기 종료
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e+1) % eval_every == 0:

                test_preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(test_preds, y_test)

                if early_stopping:
                    if loss < self.best_loss:
                        print(f"{e+1}에폭에서 검증 데이터에 대한 손실값: {loss:.3f}")
                        #print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print()
                        #print(f"Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f},",
                        #      f"\nusing the model from epoch {e+1-eval_every}")
                        print(f"{e+1}에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 {e+1-eval_every}",
                              f"에폭까지 학습된 모델에서 계산된 {self.best_loss:.3f}이다.")
                        self.net = last_model
                        # ensure self.optim is still updating self.net
                        setattr(self.optim, 'net', self.net)
                        break

                else:
                    #print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                    print(f"{e+1}에폭에서 검증 데이터에 대한 손실값: {loss:.3f}")

            if self.optim.final_lr:
                self.optim._decay_lr()
