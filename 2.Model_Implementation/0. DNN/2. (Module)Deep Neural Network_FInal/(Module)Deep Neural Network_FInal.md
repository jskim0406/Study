# Deep NeuralNetwork _ from scratch

```
[Option]

1. Number of neurons (hidden layer, output layer)
2. Loss function
    - Mean Squard Error
    - Softmax Cross Entorpy
3. Activation function
    - Linear
    - Sigmoid
    - ReLU
    - Tanh
4. Optimizer
    - SGD
    - SGD with momentum
5. Weight initializer
6. Learning rate decay
7. Dropout
```


```python
import jskim_DNN
from jskim_DNN.layers import Dense
from jskim_DNN.losses import SoftmaxCrossEntropy, MeanSquaredError
from jskim_DNN.optimizers import Optimizer, SGD, SGDMomentum
from jskim_DNN.activations import Sigmoid, Tanh, Linear, ReLU
from jskim_DNN.network import NeuralNetwork
from jskim_DNN.train import Trainer
from jskim_DNN.utils.np_utils import softmax
from jskim_DNN.utils import mnist
```

**평가기준**


```python
def mae(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    신경망 모델의 평균절대오차 계산
    '''    
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    신경망 모델의 제곱근 평균제곱오차 계산
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model: NeuralNetwork,
                          X_test: np.ndarray,
                          y_test: np.ndarray):
    '''
    신경망 모델의 평균절대오차 및 제곱근 평균제곱오차 계산
    Compute mae and rmse for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("평균절대오차: {:.2f}".format(mae(preds, y_test)))
    print()
    print("제곱근 평균제곱오차 {:.2f}".format(rmse(preds, y_test)))
```


```python
lr = NeuralNetwork(
    layers=[Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

nn = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

dl = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)
```

# 1. Boston dataset _ Regression

**데이터 로드, 테스트 / 학습 데이터 분할**


```python
from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names
```


```python
# 데이터 축척 변환
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)
```


```python
def to_2d_np(a: np.ndarray, 
          type: str="col") -> np.ndarray:
    '''
    1차원 텐서를 2차원으로 변환
    '''

    assert a.ndim == 1, \
    "입력된 텐서는 1차원이어야 함"
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

# 목푯값을 2차원 배열로 변환
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)
```

**3가지 모델 학습**


```python
# 헬퍼 함수

def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]
```


```python
trainer = Trainer(lr, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(lr, X_test, y_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 30.293
    20 에폭에서 검증 데이터에 대한 손실값: 28.469
    30 에폭에서 검증 데이터에 대한 손실값: 26.293
    40 에폭에서 검증 데이터에 대한 손실값: 25.541
    50 에폭에서 검증 데이터에 대한 손실값: 25.087
    
    평균절대오차: 3.52
    
    제곱근 평균제곱오차 5.01



```python
trainer = Trainer(nn, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(nn, X_test, y_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 27.435
    20 에폭에서 검증 데이터에 대한 손실값: 21.839
    30 에폭에서 검증 데이터에 대한 손실값: 18.918
    40 에폭에서 검증 데이터에 대한 손실값: 17.195
    50 에폭에서 검증 데이터에 대한 손실값: 16.215
    
    평균절대오차: 2.60
    
    제곱근 평균제곱오차 4.03



```python
trainer = Trainer(dl, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(dl, X_test, y_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 44.143
    20 에폭에서 검증 데이터에 대한 손실값: 25.278
    30 에폭에서 검증 데이터에 대한 손실값: 22.339
    40 에폭에서 검증 데이터에 대한 손실값: 16.500
    50 에폭에서 검증 데이터에 대한 손실값: 14.655
    
    평균절대오차: 2.45
    
    제곱근 평균제곱오차 3.83


# 2. MNIST digit _ Classification


```python
import sys
# 예제 파일 경로로 수정한 다음 주석 해제
sys.path.append(r'/Users/kimjeongseob/Desktop/Study/2.Model_Implementation/1. DNN')
```


```python
mnist.init() # 최초 실행시 주석 해제, 이후 다시 주석 처리할 것
```

    Downloading train-images-idx3-ubyte.gz...
    Downloading t10k-images-idx3-ubyte.gz...
    Downloading train-labels-idx1-ubyte.gz...
    Downloading t10k-labels-idx1-ubyte.gz...
    Download complete.
    Save complete.



```python
X_train, y_train, X_test, y_test = mnist.load()
```


```python
num_labels = len(y_train)
num_labels
```




    60000




```python
# 원-핫 인코딩
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1
```

### 데이터 정규화 : 평균 0 분산 1


```python
X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
```


```python
np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test)
```




    (-33.318421449829934,
     221.68157855017006,
     -33.318421449829934,
     221.68157855017006)




```python
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)
```


```python
np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test)
```




    (-0.424073894391566, 2.821543345689335, -0.424073894391566, 2.821543345689335)




```python
def calc_accuracy_model(model, test_set):
    return print(f'''모델 검증을 위한 정확도: {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')
```

## 1. Activation / Loss function

### 1) Sigmoid + MSE


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Sigmoid())],
            loss = MeanSquaredError(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 0.611
    20 에폭에서 검증 데이터에 대한 손실값: 0.428
    30 에폭에서 검증 데이터에 대한 손실값: 0.389
    40 에폭에서 검증 데이터에 대한 손실값: 0.374
    50 에폭에서 검증 데이터에 대한 손실값: 0.366
    
    모델 검증을 위한 정확도: 72.58%


### 2) Sigmoid + CrossEntropy


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Sigmoid()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 130,
            eval_every = 1,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)
```

    1 에폭에서 검증 데이터에 대한 손실값: 1.285
    2 에폭에서 검증 데이터에 대한 손실값: 0.970
    3 에폭에서 검증 데이터에 대한 손실값: 0.836
    4 에폭에서 검증 데이터에 대한 손실값: 0.763
    5 에폭에서 검증 데이터에 대한 손실값: 0.712
    6 에폭에서 검증 데이터에 대한 손실값: 0.679
    7 에폭에서 검증 데이터에 대한 손실값: 0.651
    8 에폭에서 검증 데이터에 대한 손실값: 0.631
    9 에폭에서 검증 데이터에 대한 손실값: 0.617
    10 에폭에서 검증 데이터에 대한 손실값: 0.599
    11 에폭에서 검증 데이터에 대한 손실값: 0.588
    12 에폭에서 검증 데이터에 대한 손실값: 0.576
    13 에폭에서 검증 데이터에 대한 손실값: 0.568
    14 에폭에서 검증 데이터에 대한 손실값: 0.557
    15 에폭에서 검증 데이터에 대한 손실값: 0.550
    16 에폭에서 검증 데이터에 대한 손실값: 0.544
    17 에폭에서 검증 데이터에 대한 손실값: 0.537
    18 에폭에서 검증 데이터에 대한 손실값: 0.533
    19 에폭에서 검증 데이터에 대한 손실값: 0.529
    20 에폭에서 검증 데이터에 대한 손실값: 0.523
    21 에폭에서 검증 데이터에 대한 손실값: 0.517
    22 에폭에서 검증 데이터에 대한 손실값: 0.512
    23 에폭에서 검증 데이터에 대한 손실값: 0.507
    24에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 23에폭까지 학습된 모델에서 계산된 0.507이다.
    
    모델 검증을 위한 정확도: 91.04%


### 3) ReLU + CrossEntropy


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=ReLU()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 5.955
    20에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 10에폭까지 학습된 모델에서 계산된 5.955이다.
    
    모델 검증을 위한 정확도: 76.38%


### 4) Tanh + CrossEntropy


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 0.630
    20 에폭에서 검증 데이터에 대한 손실값: 0.574
    30 에폭에서 검증 데이터에 대한 손실값: 0.549
    40 에폭에서 검증 데이터에 대한 손실값: 0.546
    50에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 40에폭까지 학습된 모델에서 계산된 0.546이다.
    
    모델 검증을 위한 정확도: 91.01%


## 2. Optimizer 

### 1) SGD momentum + Linear activation


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Sigmoid()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optim = SGDMomentum(0.1, momentum=0.9)

trainer = Trainer(model, SGDMomentum(0.1, momentum=0.9))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 1,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)
```

    1 에폭에서 검증 데이터에 대한 손실값: 0.615
    2 에폭에서 검증 데이터에 대한 손실값: 0.489
    3 에폭에서 검증 데이터에 대한 손실값: 0.444
    4에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 3에폭까지 학습된 모델에서 계산된 0.444이다.
    모델 검증을 위한 정확도: 92.12%


### 2) SGD momentum + Sigmoid activation


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optim = SGD(0.1)

optim = SGDMomentum(0.1, momentum=0.9)

trainer = Trainer(model, SGDMomentum(0.1, momentum=0.9))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 0.441
    20 에폭에서 검증 데이터에 대한 손실값: 0.351
    30 에폭에서 검증 데이터에 대한 손실값: 0.345
    40 에폭에서 검증 데이터에 대한 손실값: 0.338
    50에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 40에폭까지 학습된 모델에서 계산된 0.338이다.
    모델 검증을 위한 정확도: 95.51%


## 3. Learning-rate Decay

- Exponential
- Linear

### 1) Linear decay


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optimizer = SGDMomentum(0.15, momentum=0.9, final_lr = 0.05, decay_type='linear')

trainer = Trainer(model, optimizer)
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 0.419
    20 에폭에서 검증 데이터에 대한 손실값: 0.340
    30 에폭에서 검증 데이터에 대한 손실값: 0.329
    40 에폭에서 검증 데이터에 대한 손실값: 0.300
    50에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 40에폭까지 학습된 모델에서 계산된 0.300이다.
    모델 검증을 위한 정확도: 95.67%


### 2) Exponential decay


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optimizer = SGDMomentum(0.2, 
                        momentum=0.9, 
                        final_lr = 0.05, 
                        decay_type='exponential')

trainer = Trainer(model, optimizer)
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)
```

    10 에폭에서 검증 데이터에 대한 손실값: 2.870
    20에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 10에폭까지 학습된 모델에서 계산된 2.870이다.
    모델 검증을 위한 정확도: 67.39%


## 4. Weight initialize

- glorot method


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh(),
                  weight_init="glorot"),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optimizer = SGDMomentum(0.15, momentum=0.9, final_lr = 0.05, decay_type='linear')

trainer = Trainer(model, optimizer)
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 50,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True)

calc_accuracy_model(model, X_test)
```

    10에폭에서 검증 데이터에 대한 손실값: 0.373
    20에폭에서 검증 데이터에 대한 손실값: 0.289
    30에폭에서 검증 데이터에 대한 손실값: 0.271
    40에폭에서 검증 데이터에 대한 손실값: 0.269
    
    50에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 40 에폭까지 학습된 모델에서 계산된 0.269이다.
    모델 검증을 위한 정확도: 96.28%


## 5. Dropout



```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 50,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True);

calc_accuracy_model(model, X_test)
```

    10에폭에서 검증 데이터에 대한 손실값: 2.523
    
    20에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 10 에폭까지 학습된 모델에서 계산된 2.523이다.
    모델 검증을 위한 정확도: 65.95%


**Dropout 미실시**


```python
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh(),
                  weight_init="glorot",
                  ),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 50,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True);

calc_accuracy_model(model, X_test)
```

    10에폭에서 검증 데이터에 대한 손실값: 3.631
    
    20에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 10 에폭까지 학습된 모델에서 계산된 3.631이다.
    모델 검증을 위한 정확도: 63.11%

