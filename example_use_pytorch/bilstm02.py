# import random
# alltrain=[]
# for i in range(10000):
#     a=random.uniform(1, 5)
#     b=random.uniform(1, 5)variable-pass internal grinding for engineering ceramics
#     c=random.uniform(1, 5)
#     alltrain.append([a,b,c,a*3+b*6+c*8,a+b+c])
# import pandas as pd
# df=pd.DataFrame(alltrain)
#
# df.to_csv("make_csv.csv",header=True,index=False,encoding="utf-8")
# # make_data

import pandas as pd

df=pd.read_csv("data_02.csv")

train=df[df.columns[:3]]
y_train=df[df.columns[3:]]
# train.shape,y_train.shape

from sklearn.model_selection import train_test_split
train_x,Test_x, train_y,Test_y = train_test_split(train.values.reshape(-1,1,3), y_train.values, test_size=0.4, random_state=42)
val_x,test_x, val_y,test_y = train_test_split(Test_x, Test_y, test_size=0.5, random_state=42)

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_grid={'bidirectional':[True,False],'hidden_size':[16,32,64,128,256,512],
            'num_layers':[1,2,3,4]}

# Bilstm model
import torch.nn as nn
import torch
from torch.autograd import *
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt


class BiLSTMNet(nn.Module):

    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,

        )
        self.out = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x.view(len(x), 1, -1))  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        # print(out.shape)
        return out


def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

net = BiLSTMNet(test_x.shape[-1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)

# start training
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
    from sklearn.metrics import mean_squared_error

    print ("epoch :{} train_loss :{}  val_loss:{}".format(e,avgloss,val_avgloss))
    from sklearn.metrics import mean_absolute_error
    var_x = ToVariable(val_x)
    var_y = ToVariable(val_y)
    out = net(var_x)
    mse = mean_squared_error(var_y, out.detach().numpy())
    mae = mean_absolute_error(var_y, out.detach().numpy())
    msel.append(mse)
    mael.append(mae)
    print ("val mse:{} val mae:{}".format(mse,mae))

# 保存和加载
import joblib
joblib.dump(net,"bilstm02_model.joblib")
net=joblib.load("bilstm02_model.joblib")


# predict
def pre(x):
    x=x.reshape(-1,1,3)
    var_x = ToVariable(x)
    out = net(var_x)
    return out.detach().numpy()

pre(test_x)

# 回归模型评价
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
var_x = ToVariable(train_x)
var_y = ToVariable(train_y)
out = net(var_x)
mse_t = mean_squared_error(var_y, out.detach().numpy())
mae_t = mean_absolute_error(var_y, out.detach().numpy())
print("train mse:{} train mae:{} ".format(mse_t, mae_t))
var_x = ToVariable(test_x)
var_y = ToVariable(test_y)
out = net(var_x)
mse = mean_squared_error(var_y, out.detach().numpy())
mae = mean_absolute_error(var_y, out.detach().numpy())
EVRS= explained_variance_score(var_y,out.detach().numpy())
R2_S=r2_score(var_y,out.detach().numpy())
print("test mse:{} test mae:{} test EVRS:{} test R2_S:{}".format(mse, mae,EVRS,R2_S))






# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(msel)), msel,c='red')
plt.title("each epoch mse", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('mse', fontsize=20)
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(mael)), mael,c='green')
plt.title("each epoch mae", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('mae', fontsize=20)
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(lossl)), lossl,c='green')
plt.plot(range(len(vallossl)), vallossl,c='red')
plt.title("each epoch loss", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.legend(['loss','val_loss'], fontsize=20)
plt.show()




