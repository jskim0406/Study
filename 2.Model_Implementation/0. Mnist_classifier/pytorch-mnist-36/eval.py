## 라이브러리 추가하기
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

## 1. 트레이닝을 위한 하이퍼 파라미터 설정 (learning rate, batch size, epoch number)
lr = 1e-3
batch_size = 128
# num_epoch = 5

#### ckpt_dir : 체크포인트 디렉토리(학습된 모델이 저장될 디렉토리)
ckpt_dir = './drive/My Drive/Colab Notebooks/0. Mnist_classifier/pytorch-mnist-36/checkpoint'

#### log_dif : 로그 디렉토리(텐서보드 파일들이 저장될 디렉토리)
log_dir = './drive/My Drive/Colab Notebooks/0. Mnist_classifier/pytorch-mnist-36/log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 2. 네트워크를 구축하기
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1,320)

        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)

        return x


## 3. 네트워크를 저장하거나 불러올 수 있는 함수 구현
### save(), load()
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(), 'optim':optim.state_dict()},
               './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir,ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim

## 4. MNIST 데이터셋 불러오기
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

dataset = datasets.MNIST(download=True, root='./', train=False, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_data = len(loader.dataset)
num_batch = np.ceil(num_data/batch_size)

## 5. 네트워크 및 손실함수 설정
net = Net().to(device)
params = net.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params, lr=lr)

writer = SummaryWriter(log_dir=log_dir)

# training 후 가장 마자막 model, optimizer load
net, optim = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

## 6. 네트워크 학습을 위한 for 문 구현
# for epoch in range(1, num_epoch+1):
with torch.no_grad():
    # net.train()
    net.eval()

    loss_arr = []
    acc_arr = []

    for batch, (input, label) in enumerate(loader,1):
        input = input.to(device)
        label = label.to(device)

        output = net(input)
        pred = fn_pred(output)

        # optim.zero_grad()

        loss = fn_loss(output, label)
        acc = fn_acc(pred, label)

        # loss.backward()

        # optim.step()

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]

        print('Test : BATCH %04d/%04d | LOSS %.4f | ACC %.4f' %
            (batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))
#
#     writer.add_scalar('loss', np.mean(loss_arr), epoch)
#     writer.add_scalar('acc', np.mean(acc_arr), epoch)
#
#     save(ckpt_dir = ckpt_dir, net = net, optim = optim, epoch = epoch)
#
# writer.close()


