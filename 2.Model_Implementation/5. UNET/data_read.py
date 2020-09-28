# data : 512 512 30 (30개의 frame으로 이루어져있음)
# 30개의 frame을 하나씩 나누어 사용할 수 있도록 함.

## 필요한 라이브러리 import
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'
name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data,name_label))
img_input = Image.open(os.path.join(dir_data,name_input))

ny, nx = img_label.size  # 512, 512
nframe = img_label.n_frames  # 30


## train, valid, test set 분리
nframe_train = 24
nframe_val = 3
nframe_test = 3

### train, valid, test set 저장될 디렉토리 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

### 실제 데이터 분리 후 저장
### 1. frame을 랜덤으로 저장하기 위해, frame의 index를 shuffle
id_frame = np.arange(nframe)   # [0,1,2,3, ... ,29]
np.random.shuffle(id_frame)   # [ shuffled ]

### 2. 데이터셋 저장
offset_nframe=0
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])   # img_label의 프레임 별로 select
    img_input.seek(id_frame[i + offset_nframe])   # img_input의 프레임 별로 select

    label_ = np.asarray(img_label)   # img_label의 프레임을 ndarray로 생성
    input_ = np.asarray(img_input)   # img_input의 프레임을 ndarray로 생성

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)  # ndarray로 frame별로 저장
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)  # ndarray로 frame별로 저장

offset_nframe += nframe_train
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

offset_nframe += nframe_val
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

### 3. 생성된 train, valid, test data 확인(출력)
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title("label")

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title("input")

plt.show()

##

