import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
#定义一些超参数
import net
batch_size=64
learning_rate=1e-2
num_epoches=20
#预处理
data_tf=transforms.Compose(
[transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])#将图像转化成tensor，然后继续标准化，就是减均值，除以方差

#读取数据集
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)
#使用内置的函数导入数据集
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#导入网络，定义损失函数和优化方法
model=net.simpleNet(28*28,300,100,10)
if torch.cuda.is_available():#是否使用cuda加速
    model=model.cuda()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)
import net
n_epochs=5
for epoch in range(n_epochs):
    running_loss=0.0
    running_correct=0
    print("epoch {}/{}".format(epoch,n_epochs))
    print("-"*10)
    for data in train_loader:
        img,label=data
        img=img.view(img.size(0),-1)
        if torch.cuda.is_available():
            img=img.cuda()
            label=label.cuda()
        else:
            img=Variable(img)
            label=Variable(label)
        out=model(img)#得到前向传播的结果
        loss=criterion(out,label)#得到损失函数
        print_loss=loss.data.item()
        optimizer.zero_grad()#归0梯度
        loss.backward()#反向传播
        optimizer.step()#优化
        running_loss+=loss.item()
        epoch+=1
        if epoch%50==0:
            print('epoch:{},loss:{:.4f}'.format(epoch,loss.data.item()))
