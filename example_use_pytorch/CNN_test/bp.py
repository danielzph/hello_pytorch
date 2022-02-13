import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# 读数据
df=pd.read_csv("test.csv")
train=df[df.columns[:8]]
y_train=df[df.columns[8:15]]


train=train.values.reshape(-1,10,8)
y_train=y_train.values[:40]

# print(train)
#
# print(y_train)


import numpy as np
# train=train[:,np.newaxis,:,:]       #分场合



# 拆分数据集
train_x,test_x,val_x = train[:20],train[20:30],train[30:40]
train_y,test_y,val_y=y_train[:20],y_train[20:30],y_train[30:40]



# class test_BP(nn.Module):
#     def __init__(self):
#         super(test_BP, self).__init__()
#         self.fc1 = nn.Linear(8, 32)
#         self.fc2 = nn.Linear(32, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 5)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x



# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_output=6):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(128, num_output)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

import warnings
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=6):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3),  # 64,10,10
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        #
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3),  # 128,8,8
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveMaxPool2d((4,4)))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x


class BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=6):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        self.input_size = 8
        self.num_layers = 2
        self.V = 10
        # self.embed1 = nn.Sequential(
        #     nn.Conv1d(in_channel, self.kernel_num, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(self.kernel_num),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2))
        # self.embed2 = nn.Sequential(
        #     nn.Conv1d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(self.kernel_num*2),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveMaxPool1d(self.V))
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size,
                              num_layers=self.num_layers, bidirectional=True,
                              batch_first=True, bias=False)
        self.linear1 = nn.Sequential(nn.Linear(self.V * 2 * self.hidden_size, self.hidden_size * 4), nn.ReLU(),
                                     nn.Dropout())
        self.linear2 = nn.Linear(self.hidden_size * 4, out_channel)


    def forward(self, x):
        # x = self.embed1(x)
        # x = self.embed2(x)
        x = x.view(-1, self.input_size, self.V)
        x = torch.transpose(x, 1, 2)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = bilstm_out.contiguous().view(bilstm_out.size(0), -1)
        logit = self.linear1(bilstm_out)
        logit = self.linear2(logit)

        return logit


class BiLSTMNet(nn.Module):

    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            bidirectional=True,

        )
        self.out = nn.Sequential(
            nn.Linear(128, 6)
        )
        self.h0 = torch.randn(2, 3, 20)
        self.c0 = torch.randn(2, 3, 20)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x.view(len(x), 1, -1),(self.h0, self.c0))  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        print(out.shape)
        return out




class CNN1d_BiLSTM(nn.Module):
    def __init__(self, in_channel=10, out_channel=6):
        super(CNN1d_BiLSTM, self).__init__()
        self.hidden_size = 64
        self.input_size = 8
        self.num_layers = 2
        self.V = 10
        self.embed1 = nn.Sequential(
            nn.Conv1d(in_channel,out_channels=self.V , kernel_size=3, padding=1),
            nn.BatchNorm1d(self.V),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # self.embed2 = nn.Sequential(
        #     nn.Conv1d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(self.kernel_num*2),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveMaxPool1d(self.V))
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size,
                              num_layers=self.num_layers, bidirectional=True,
                              batch_first=True, bias=False)
        self.linear1 = nn.Sequential(nn.Linear(  2* self.hidden_size, self.hidden_size), nn.ReLU(),
                                     nn.Dropout())
        self.linear2 = nn.Linear(self.hidden_size, out_channel)


    def forward(self, x):
        x = self.embed1(x)
        # x = self.embed2(x)
        x = x.view(-1, self.input_size, self.V)
        x = torch.transpose(x, 1, 2)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = bilstm_out.view(bilstm_out.size(0), -1)
        logit = self.linear1(bilstm_out)
        logit = self.linear2(logit)

        return logit





def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# net = test_BP()
# net = ConvNet(num_output=6).to(device)
# net = ConvNet(num_output=6)
# net = CNN(in_channel=1, out_channel=6)
net = BiLSTM(in_channel=1, out_channel=6)
# net = BiLSTMNet(test_x.shape[-1])     # 这个还没写出来
# net = CNN1d_BiLSTM(in_channel=10, out_channel=6)     # 这个还没写出来



criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)



# 训练
epoch=500
batchsize=2
msel=[]
mael=[]
lossl=[]
vallossl=[]
for e in range(epoch):
    avgloss=0
    for i in range(batchsize,len(train_x),batchsize):
            var_x = ToVariable(train_x[i-batchsize:i])
            var_y = ToVariable(train_y[i-batchsize:i])
            # forwardss
            out = net(var_x)
            loss = criterion(out, var_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            avgloss+=loss.data/len(train_x)
            optimizer.step()
    lossl.append(avgloss)
    val_avgloss=0
    for i in range(batchsize,len(val_x),batchsize):
            var_x = ToVariable(val_x[i-batchsize:i])
            var_y = ToVariable(val_y[i-batchsize:i])
            out = net(var_x)
            loss = criterion(out, var_y)
            val_avgloss+=loss.data/len(val_x)
    vallossl.append(val_avgloss)
    print ("epoch :{} train_loss :{}  val_loss:{}".format(e,avgloss,val_avgloss))

    var_x = ToVariable(val_x)
    var_y = ToVariable(val_y)
    out = net(var_x)
    mse = mean_squared_error(var_y, out.detach().numpy())
    mae = mean_absolute_error(var_y, out.detach().numpy())
    msel.append(mse)
    mael.append(mae)
    print ("val mse:{} val mae:{}".format(mse,mae))

# 保存和加载
# joblib.dump(net,"bp_model.joblib")
# net=joblib.load("bp_model.joblib")

# predict
def pre(x):
    # x=x.reshape(-1,1,8)
    var_x = ToVariable(x)
    out = net(var_x)
    return out.detach().numpy()

pre(test_x)

# 回归模型评价
var_x = ToVariable(test_x)
var_y = ToVariable(test_y)
out = net(var_x)
mse = mean_squared_error(var_y, out.detach().numpy())
mae = mean_absolute_error(var_y, out.detach().numpy())
EVRS= explained_variance_score(var_y,out.detach().numpy())
R2_S=r2_score(var_y,out.detach().numpy())
print("test mse:{} test mae:{} test EVRS:{} test R2_S:{}".format(mse, mae,EVRS,R2_S))



# plot
plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(msel)), msel,c='red')
plt.title("each eooch mse", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('mse', fontsize=20)
plt.show()


plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(mael)), mael,c='green')
plt.title("each eooch mae", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('mae', fontsize=20)
plt.show()


plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(lossl)), lossl,c='green')
plt.plot(range(len(vallossl)), vallossl,c='red')
plt.title("each eooch loss", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.legend(['loss','val_loss'], fontsize=20)
plt.show()




