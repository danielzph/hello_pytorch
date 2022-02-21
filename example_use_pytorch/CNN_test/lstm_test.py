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
df=pd.read_csv("f_data06.csv")
# train=df[df.columns[0:5]]
y_train=df[df.columns[18]]

df2=pd.read_csv("../../data/shiyan/f_ae.csv")
train=df2[df2.columns[0:14]]



train=train.values.reshape(-1,3,14)
# train=train[:30]
y_train=y_train.values[:198]


# print(train)
#
# print(y_train)


import numpy as np
# train=train[:,np.newaxis,:,:]       #分场合
y_train=y_train[:,np.newaxis]


# 拆分数据集
# train_x,test_x,val_x = train[:70],train[70:85],train[85:99]
# train_y,test_y,val_y=y_train[:70],y_train[70:85],y_train[85:99]
train_x,Test_x, train_y,Test_y = train_test_split(train, y_train, test_size=2/9, random_state=2)
val_x,test_x, val_y,test_y = train_test_split(Test_x, Test_y, test_size=0.5, random_state=2)


class BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        self.input_size = 14
        self.num_layers = 2
        self.V = 3
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


def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)



net = BiLSTM(in_channel=1, out_channel=1)



criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.002, weight_decay=0.0001)



# 训练
epoch=500
batchsize=10
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










