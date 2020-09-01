import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import copy
import net
import pickle
import os

from torch import nn,optim
from torch.autograd import Variable
#from torch.utils.data import DataLoader
from torchvision import datasets,transforms



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
learning_rate = 0.001
n_epochs = 300
n_hidden_1 = 50
n_hidden_2 = 50
out_dim = 8
splitlen = 5120


#预处理
data_tf=transforms.Compose(
[transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])#将图像转化成tensor，然后继续标准化，就是减均值，除以方差

#读取数据集
train_dataset = load_pkl('train_data.pkl')
train_data_array = []
for n in range(6400):
    train_data_array.append(Inserting_zeros(train_dataset[n]))

a = np.load('finalLabelsTrain.npy')
b = np.array([i-1 for i in a])




#处理数据集
train_dataset = train_data_array[0:splitlen]
train_dataset = torch.Tensor(train_dataset)
train_label = b[0:splitlen]
train_label = torch.LongTensor(train_label)
test_dataset = train_data_array[splitlen:6400]
test_dataset = torch.Tensor(test_dataset)
test_label = b[splitlen:6400]
test_label = torch.LongTensor(test_label)

# 先转换成 torch 能识别的 Dataset
train_dataset_all = Data.TensorDataset(train_dataset, train_label)
test_dataset_all = Data.TensorDataset(test_dataset, test_label)

#使用内置的函数导入数据集
train_loader = Data.DataLoader(dataset = train_dataset_all, batch_size = batch_size,shuffle = True)
test_loader = Data.DataLoader(dataset = test_dataset_all,batch_size = batch_size,shuffle = False)

#导入网络，定义损失函数和优化方法
model = net.simpleNet(64*64,n_hidden_1,n_hidden_2,out_dim)
if torch.cuda.is_available():#是否使用cuda加速
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.5)
loss_track = np.zeros(n_epochs)
loss_check = np.inf

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    epoch_sub = 0
    print("epoch {}/{}".format(epoch,n_epochs))
    print("-"*10)
    for data in train_loader:
        img,label = data
        img = img.view(img.size(0),-1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)#得到前向传播的结果
        loss = criterion(out,label)#得到损失函数
        print_loss = loss.data.item()
        loss_track[epoch] = loss.detach().cpu().numpy()
        optimizer.zero_grad()#归0梯度
        loss.backward()#反向传播
        optimizer.step()#优化
        running_loss += loss.item()
        epoch_sub += 1

        if loss < loss_check:
            loss_check = loss
            best_model_wts = copy.deepcopy(model.state_dict())
        if epoch_sub%50 == 0:
            print('epoch:{},loss:{:.4f}'.format(epoch_sub,loss.data.item()))


model.load_state_dict(best_model_wts)


#测试网络
model.eval()#将模型变成测试模式
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img,label = data
    img = img.view(img.size(0),-1)#测试集不需要反向传播，所以可以在前项传播的时候释放内存，节约内存空间
    if torch.cuda.is_available():
        img = Variable(img,volatile = True).cuda()
        label = Variable(label,volatile = True).cuda()
    else:
        img = Variable(img,volatile = True)
        label = Variable(label,volatile = True)
    out = model(img)
    loss = criterion(out,label)
    eval_loss += loss.item()*label.size(0)
    _,pred = torch.max(out,1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('test loss:{:.6f},ac:{:.6f}'.format(eval_loss/(len(test_dataset)),eval_acc/(len(test_dataset))))

a=1