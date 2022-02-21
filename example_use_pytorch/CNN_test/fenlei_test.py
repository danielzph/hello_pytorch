import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy as np
from sklearn import svm

# 读数据
df=pd.read_csv("f_data06.csv")
# train=df[df.columns[5:6]]
# train=train[:4356]
y_train=df[df.columns[19]]

df2=pd.read_csv("../../data/shiyan/f_ae.csv")
train=df2[df2.columns[0:5]]



train=train.values.reshape(-1,3,5)
# train=train.reshape(-1,22*1)
# train=train.values.reshape(-1,22)
# train=train[:,np.newaxis,:,:]
train = torch.tensor(train, dtype=torch.float)
y_train=y_train.values[:198]
y_train=torch.tensor(y_train, dtype=torch.long)
# labels=[]
# for v in range(0,len(y_train)):
#     labels.append(int(y_train[v]))



train_x,Test_x, train_y,Test_y = train_test_split(train, y_train, test_size=3/9, random_state=2)
val_x,test_x, val_y,test_y = train_test_split(Test_x, Test_y, test_size=0.5, random_state=2)

train_dataset = Data.TensorDataset(train_x, train_y)
train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_dataset = Data.TensorDataset(val_x, val_y)
val_dataloader = Data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
test_dataset = Data.TensorDataset(test_x, test_y)
test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)





def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

class BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        self.input_size = 5
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

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self,  in_channel,out_channel):
        super(ConvNet, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1=nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=7),  # 16, 26 ,26
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True))

        self.fc = nn.Linear(64, out_channel)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class CNN(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Linear(416, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.fc(x)

        return x





# 模型实例化
net = BiLSTM(in_channel=1, out_channel=2)
# net = ConvNet(in_channel=1, out_channel=2)
# net = CNN(in_channel=1, out_channel=2)

# criterion = nn.MSELoss()
# criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()


# 训练
epoch=500
loss_l=[]
loss_val_l=[]
Val_accuracy=[]
for e in range(epoch):
    epoch_loss = 0
    epoch_loss_val=0
    epoch_acc = 0
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data
    print('Epoch [{}/{}],  Loss: {:.4f}'.format(e + 1, epoch, epoch_loss.item()/(i+1)))
    correct = torch.zeros(1).squeeze().cpu()
    total = torch.zeros(1).squeeze().cpu()
    loss_l.append(epoch_loss.item()/i)
    for i, (x, y) in enumerate(val_dataloader):
        out = net(x)
        loss_val = criterion(out, y)
        # loss_val.backward()
        # optimizer.step()
        prediction = torch.argmax(out, 1)
        correct += (prediction == y).cpu().sum().float()
        total += len(y)
        epoch_loss_val += loss_val.data
    val_accuracy = (correct / total).cpu().detach().data.numpy()
    loss_val_l.append(epoch_loss_val.item()/(i+1))
    Val_accuracy.append(val_accuracy)
    print('loss_val:', epoch_loss_val.item()/(i+1))
    print('val_accuracy:',val_accuracy)

# 测试模型
correct_test=0
total_test=0
epoch_loss_test=0
for i, (x, y) in enumerate(test_dataloader):
    out = net(x)
    loss_test = criterion(out, y)
    prediction = torch.argmax(out, 1)
    correct_test += (prediction == y).cpu().sum().float()
    total_test += len(y)
    epoch_loss_test += loss_test.data
test_accuracy = (correct_test / total_test).cpu().detach().data.numpy()
print('test_accuracy:', test_accuracy)




plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(loss_l)), loss_l,c='red')
plt.title("each eooch lose", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('lose', fontsize=20)
plt.show()

plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(loss_val_l)), loss_val_l,c='green')
plt.title("each eooch val loss", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('val loss', fontsize=20)
plt.show()


plt.figure(figsize=(24,8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(range(len(Val_accuracy)), Val_accuracy,c='black')
plt.title("each eooch Val_accuracy", fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('Val_accuracy', fontsize=20)
plt.show()




# # 支持向量机 对比
# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
# clf.fit(train_x,train_y)
# print( clf.score(train_x, train_y) ) # 精度
# print(clf.score(val_x, val_y))
# print(clf.score(test_x, test_y))
#
# # 随机森林 对比
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(train_x, train_y)
# print(classifier.score(train_x, train_y))
# print(classifier.score(val_x, val_y))
# print(classifier.score(test_x, test_y))
# y_pred = classifier.predict(test_x)
# cm = confusion_matrix(test_y, y_pred)
# df=pd.DataFrame(cm,index=["defective", "good"],columns=["defective", "good"])
# sns.heatmap(df,annot=True)
# # plt.show()
#
# # 决策树 对比
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
# clf.fit(train_x, train_y)
# print(clf.score(train_x, train_y) ) # 精度
# print(clf.score(val_x, val_y))
# print(clf.score(test_x, test_y))
#
#
# # KNN 对比
# from sklearn import neighbors
# knn = neighbors.KNeighborsClassifier()#KNN模型
# knn.fit(train_x, train_y)
# print(knn.score(train_x, train_y) ) # 精度
# print(knn.score(val_x, val_y))
# print(knn.score(test_x, test_y))

