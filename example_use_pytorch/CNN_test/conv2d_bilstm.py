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


# df2=pd.read_csv("../../data/shiyan/f_ae02.csv")
# train=df2[df2.columns[0:7]]

train=train.values.reshape(-1,22,14)
y_train=y_train.values[:198]

# print(train)
#
# print(y_train)


import numpy as np
train=train[:,np.newaxis,:,:]       #分场合
y_train=y_train[:,np.newaxis]


# 拆分数据集
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



class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=2):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

class channelAttention(nn.Module):
    def __init__(self , in_planes , ration = 4):
        super(channelAttention, self).__init__()

        '''
        AdaptiveAvgPool2d():自适应平均池化
                            不需要自己设置kernelsize stride等
                            只需给出输出尺寸即可
        '''

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 通道数不变，H*W变为1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1) #

        self.fc1 = nn.Conv2d(in_planes , in_planes // ration , 1 , bias = False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//ration , in_planes ,1, bias = False)

        self.sigmoid = nn.Sigmoid()

    def forward(self , x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        #print(avg_out.shape)
        #两层神经网络共享
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print(avg_out.shape)
        # print(max_out.shape)
        out = avg_out + max_out
        # print(out.shape)
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)




class Covn2D_BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Covn2D_BiLSTM, self).__init__()
        self.hidden_dim = 16
        self.kernel_num = 16
        self.num_layers = 1
        self.V = 5
        self.embed1 = nn.Sequential(
            nn.Conv2d(in_channel, self.kernel_num, kernel_size=3,padding=1),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool2d(6)
            )
        self.embed2 = nn.Sequential(
            nn.Conv2d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.kernel_num*2),
            nn.ReLU(inplace=True),
            # nn.AdaptiveMaxPool2d(self.V))
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.hidden2label1 = nn.Sequential(nn.Linear(15  * self.hidden_dim, self.hidden_dim * 3),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.hidden2label2 = nn.Sequential(nn.Linear(self.hidden_dim * 3, self.hidden_dim * 1),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.hidden2label3 = nn.Linear(self.hidden_dim , out_channel)
        self.bilstm = nn.LSTM(self.kernel_num*2, self.hidden_dim,
                              num_layers=self.num_layers, bidirectional=False,
                              batch_first=True, bias=False)
        self.se1=SE_Block(16,reduction=4)
        self.sa1=SpatialAttention(kernel_size=3)
        self.ca1=channelAttention(in_planes=16)


    def forward(self, x):
        x = self.embed1(x)
        x = self.se1(x)
        # x = self.ca1(x)*x
        # x = self.sa1(x)*x
        x = self.embed2(x)
        x = x.view(-1, self.kernel_num*2, 15)
        x = torch.transpose(x, 1, 2)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out) #
        bilstm_out = bilstm_out.contiguous().view(bilstm_out.size(0), -1)
        logit = self.hidden2label1(bilstm_out)
        logit = self.hidden2label2(logit)
        logit = self.hidden2label3(logit)

        return logit





def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# net = test_BP()
# net = ConvNet(num_output=6).to(device)
# net = ConvNet(num_output=6)
# net = CNN(in_channel=1, out_channel=6)
# net = BiLSTM(in_channel=1, out_channel=6)
# net = BiLSTMNet(test_x.shape[-1])     # 这个还没写出来
net=Covn2D_BiLSTM(in_channel=1, out_channel=1)



criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001)


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




plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(out.detach().numpy(),c='green')
plt.plot(test_y,c='red')
plt.title("预测结果", fontsize=20)
plt.xlabel('工件次数', fontsize=20)
plt.ylabel('直径d', fontsize=20)
plt.legend(['预测值','实际值'], fontsize=20)
plt.show()



pre_y=out.detach().numpy()*0.0513+39.128
test_y=test_y*0.0513+39.128
plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(pre_y,c='green')
plt.plot(test_y,c='red')
plt.title("预测结果", fontsize=20)
plt.xlabel('工件次数', fontsize=20)
plt.ylabel('直径d', fontsize=20)
plt.legend(['预测值','实际值'], fontsize=20)
plt.show()