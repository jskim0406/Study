##
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
import matplotlib.pyplot as plt

## parameter setting
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './drive/My Drive/Colab Notebooks/6. Unet/pytorch_unet_36/datasets'
ckpt_dir = './drive/My Drive/Colab Notebooks/6. Unet/pytorch_unet_36/checkpoint'
log_dir = './drive/My Drive/Colab Notebooks/6. Unet/pytorch_unet_36/log'  # tensorboard log 파일 저장

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## unet network
class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
            layers=[]
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Encoder part
        # first stage
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # second stage
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # third stage
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # fourth stage
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)  # 32, 32, 512 output

        # fifth stage
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024) # encoder part 마무리


        # Decoder part
        # fifth stage
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # fourth stage
        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512) # skip-connection이 있으므로, in_channels 수 주의 필요
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # third stage
        self.dec3_2 = CBR2d(in_channels=2*256, out_channels=256) # skip-connection이 있으므로, in_channels 수 주의 필요
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # second stage
        self.dec2_2 = CBR2d(in_channels=2*128, out_channels=128) # skip-connection이 있으므로, in_channels 수 주의 필요
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # first stage
        self.dec1_2 = CBR2d(in_channels=2*64, out_channels=64) # skip-connection이 있으므로, in_channels 수 주의 필요
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # 1 x 1 conv layer (class output 산출 위함)
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    # 각 레이어들 연결
    def forward(self, x):
        # encode part
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # decode part
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) # cat_dim = [0:batch, 1:channel, 2:height, 3:width]
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1) # cat_dim = [0:batch, 1:channel, 2:height, 3:width]
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)  # cat_dim = [0:batch, 1:channel, 2:height, 3:width]
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)  # cat_dim = [0:batch, 1:channel, 2:height, 3:width]
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.fc(dec1_1)

        return output

## Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [lst for lst in lst_data if lst.startswith('label')]
        lst_input = [lst for lst in lst_data if lst.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # data의 normalizing( 0~255 -> 0~1 )
        label = label/255.0
        input = input/255.0

        # dimension setting(3 dimension 갖도록. 특히, channel이 별도로 없는 경우라도, channel dimension을 꼭 세팅해줘야 함)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input' : input, 'label' : label}

        # transform
        if self.transform:
            data = self.transform(data)

        return data

## transform 구현
## 1. numpy -> tensor
class ToTensor():
    def __call__(self,data):
        label, input = data['label'], data['input']

        # Image의 numpy 차원 : (y, x, ch)
        # Image의 tensor 차원 : (ch, y, x)
        label = label.transpose(2, 0, 1).astype(np.float32) # ch의 dimension을 옮겨주기
        input = input.transpose(2, 0, 1).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
        return data

## 2. normalization
class Normalization():
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        # label은 그 자체로 class 값이기 때문에, normalization 해주면 안됨

        data = {'label': label, 'input': input}
        return data

## 3. RandomFlip : 50% 등의 확률로, up/down, left/right 데이터 회전
class RandomFlip():
    def __call__(self, data):
        input, label = data['input'], data['label']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}
        return data


## Training Network
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

dataset_train = CustomDataset(data_dir=os.path.join(data_dir,'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = CustomDataset(data_dir=os.path.join(data_dir,'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

## Network 생성
net = UNet().to(device)

## 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

## optimizer 정의
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

## 그 밖에 부수적인 variable 설정
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

## 그 밖에 부수적인 function 설정
# tensor to numpy
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
# denormalization
fn_denormal = lambda x, mean, std: (x*std) + mean
# classification
fn_class = lambda x: 1.0*(x>0.5)

## Tensorboard를 사용하기 위한, summary writer 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## network 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## network 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


## for-loop(학습)
st_epoch = 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optimizer)

for epoch in range(st_epoch+1, num_epoch+1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):

        # forward
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)
        cost = fn_loss(output, label)

        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        loss_arr += [cost.item()]

        print("Train: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS(mean per epoch) %.4f" %
              (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

        # tensorboard에 input, output, label 저장
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denormal(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        writer_train.add_image('label', label, num_batch_train * (epoch-1) + batch, dataformats='NHWC')
        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

    # loss를 tensorboard에 저장
    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)


    # validation
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)
            cost = fn_loss(output, label)

            # backward
            # optimizer.zero_grad()
            # cost.backward()
            # optimizer.step()

            loss_arr += [cost.item()]

            print("Valid: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS(mean per epoch) %.4f" %
                (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

            # tensorboard에 input, output, label 저장
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denormal(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_val.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

    # loss를 tensorboard에 저장
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

    if epoch % 10 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optimizer, epoch=epoch)


writer_train.close()
writer_val.close()










