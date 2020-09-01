from __future__ import print_function
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import os
from torch import nn,optim
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def Inserting_zeros(X):
    r,c = np.shape(X)
    if (c % 2) == 0:
        if(r % 2) == 0:
            n_peddingrow = np.floor_divide(64-r,2)           
            X = np.pad(X,((n_peddingrow,n_peddingrow),(0,0)),"constant",constant_values = (0,0))
        else:
            n_peddingrow = np.floor_divide(64-r,2)           
            X = np.pad(X,((n_peddingrow,n_peddingrow+1),(0,0)),"constant",constant_values = (0,0))
        n_peddingcol = np.floor_divide(64-c,2)    
        X = np.pad(X,((0,0),(n_peddingcol,n_peddingcol)),"constant",constant_values = (0,0))
    else:
        if(r % 2) == 0:
            n_peddingrow = np.floor_divide(64-r,2)
            X = np.pad(X,((n_peddingrow,n_peddingrow),(0,0)),"constant",constant_values = (0,0))
        else:
            n_peddingrow = np.floor_divide(64-r,2)
            X = np.pad(X,((n_peddingrow,n_peddingrow),(0,0)),"constant",constant_values = (0,0))
        n_peddingcol = np.floor_divide(64-c,2)
        X = np.pad(X,((0,0),(n_peddingcol,n_peddingcol+1)),"constant", constant_values = (0,0))
    return X


#定义一些超参数
batch_size = 32
learning_rate = 0.01
n_epochs = 200
n_hidden_1 = 10
n_hidden_2 = 20
out_dim = 8


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(1, n_hidden_1, kernel_size=5)
        self.conv2 = nn.Conv2d(n_hidden_1, n_hidden_2, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(3380, out_dim)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        return F.log_softmax(x)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = torch.unsqueeze(data,1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = torch.unsqueeze(data,1)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#读取数据集
train_dataset = load_pkl('sieved_data.pkl')
train_data_array = []
for n in range(6098):
    train_data_array.append(Inserting_zeros(train_dataset[n]))

train_label = load_pkl('sieved_label.pkl')
#train_label_array = []
#for m in range(6337):
#    train_label_array.append(Inserting_zeros(train_label[m]))

#a = np.load('finalLabelsTrain.npy')
b = np.array([i-1 for i in train_label])




#处理数据集
train_dataset, test_dataset, train_label, test_label = train_test_split(train_data_array, b, test_size=.1, random_state=0)

train_dataset = torch.Tensor(train_dataset)
train_label = torch.LongTensor(train_label)
test_dataset = torch.Tensor(test_dataset)
test_label = torch.LongTensor(test_label)

# 先转换成 torch 能识别的 Dataset
train_dataset_all = Data.TensorDataset(train_dataset, train_label)
test_dataset_all = Data.TensorDataset(test_dataset, test_label)

#使用内置的函数导入数据集
train_loader = Data.DataLoader(dataset = train_dataset_all, batch_size = batch_size,shuffle = True)
test_loader = Data.DataLoader(dataset = test_dataset_all,batch_size = batch_size,shuffle = False)

#导入网络，定义损失函数和优化方法
model = Net()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.5)

for epoch in range(1, n_epochs):
    train(epoch)
    test()

a=1