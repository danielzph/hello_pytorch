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
df=pd.read_csv("f_data07.csv")
train=df[df.columns[0:14]]
y_train=df[df.columns[18]]


train=train.values.reshape(-1,22,14)
y_train=y_train.values[:198]

# print(train)
#
# print(y_train)


import numpy as np
train=train[:,np.newaxis,:,:]       #分场合
y_train=y_train[:,np.newaxis]


# 拆分数据集
# train_x,test_x,val_x = train[:150],train[150:174],train[174:198]
# train_y,test_y,val_y=y_train[:150],y_train[150:174],y_train[174:198]
train_x,Test_x, train_y,Test_y = train_test_split(train, y_train, test_size=2/9, random_state=2)
val_x,test_x, val_y,test_y = train_test_split(Test_x, Test_y, test_size=0.5, random_state=2)



# 交叉验证
# train_x,test_x,val_x = train[:165],train[165:198],train[165:198]
# train_y,test_y,val_y=y_train[:165],y_train[165:198],y_train[165:198]
# train_x,test_x,val_x = train[33:198],train[0:33],train[0:33]
# train_y,test_y,val_y=y_train[33:198],y_train[0:33],y_train[0:33]
# train_x,test_x,val_x = np.append(train[0:33],train[66:198],axis=0),train[33:66],train[33:66]
# train_y,test_y,val_y=np.append(y_train[0:33],y_train[66:198],axis=0),y_train[33:66],y_train[33:66]
# train_x,test_x,val_x = np.append(train[0:66],train[99:198],axis=0),train[66:99],train[66:99]
# train_y,test_y,val_y=np.append(y_train[0:66],y_train[99:198],axis=0),y_train[66:99],y_train[66:99]
# train_x,test_x,val_x = np.append(train[0:99],train[132:198],axis=0),train[99:132],train[99:132]
# train_y,test_y,val_y=np.append(y_train[0:99],y_train[132:198],axis=0),y_train[99:132],y_train[99:132]
# train_x,test_x,val_x = np.append(train[0:132],train[165:198],axis=0),train[132:165],train[132:165]
# train_y,test_y,val_y=np.append(y_train[0:132],y_train[165:198],axis=0),y_train[132:165],y_train[132:165]




# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_output=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Linear(240, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, num_output)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        out = self.fc(out)
        return out


import warnings
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=2):
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
            nn.Linear(1664, 128),
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

def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# net = test_BP()
# net = ConvNet(num_output=6).to(device)
net = ConvNet(num_output=1)
# net = CNN(in_channel=1, out_channel=1)




criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=0.001)



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


plt.rcParams['font.sans-serif']=['SimHei','Times New Roman'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
# plot
plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(msel)), msel,c='red')
plt.title("每次eooch下的均方误差（MSE）", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('mse', fontsize=20)
plt.show()


plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(mael)), mael,c='green')
plt.title("每次eooch下的平均绝对值误差（MAE）", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('mae', fontsize=20)
plt.show()


plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(lossl)), lossl,c='green')
plt.plot(range(len(vallossl)), vallossl,c='red')
plt.title("每次eooch下的损失（loss）", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('损失（loss）', fontsize=20)
plt.legend(['训练集损失','验证集损失'], fontsize=20)
plt.show()


