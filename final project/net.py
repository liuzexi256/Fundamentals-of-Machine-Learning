import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

#定义三层全连接神经网络
class simpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):#输入维度，第一层的神经元个数、第二层的神经元个数，以及第三层的神经元个数
        super(simpleNet,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)
        #self.activation = nn.Tanh()
    def forward(self,x):
        #x=self.activation(self.layer1(x))
        #x=self.activation(self.layer2(x))
        #x=self.activation(self.layer3(x))
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x
    
    
#添加激活函数
class Activation_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(NeutalNetwork,self).__init__()
        self.layer1=nn.Sequential(#Sequential组合结构
        nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(
        nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3=nn.Sequential(
        nn.Linear(n_hidden_2,out_dim))
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

#添加批标准化处理模块,皮标准化放在全连接的后面，非线性的前面
class Batch_Net(nn.Module):
    def _init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Batch_net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNormld(n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNormld(n_hidden_2),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))
    def forword(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x