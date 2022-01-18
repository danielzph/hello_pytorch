import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim


# 数据预处理
# 1.自编码器温度数据同步化处理
# 1.1 读取数据
df=pd.read_csv("../../data/shiyan/210702-0937_Dist.csv")
train=df[df.columns[1:2]]
train=np.array(train)
train=train[:300]
# print(train)

tem=[]
new_train=[]
for i in range(len(train)):
    tem.append(train[i][0])
    if (i+1)%10==0:
        new_train.append(tem)
        tem=[]
# print(new_train)
train_x,test_x= train_test_split(new_train, test_size=2/8, random_state=42)

#线性自编码器
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 8),
            nn.Tanh(),
            nn.Linear(8, 5),
            nn.Tanh(),
            nn.Linear(5, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.Tanh(),
            nn.Linear(5, 8),
            nn.Tanh(),
            nn.Linear(8, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#参数设定
epochs = 500
BATCH_SIZE = 64
LR = 0.0005

# 数据类型转换
trainData = torch.FloatTensor(train_x)
testData = torch.FloatTensor(test_x)

# 构建张量数据集
train_dataset = TensorDataset(trainData, trainData)
test_dataset = TensorDataset(testData, testData)
trainDataLoader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

# 初始化
autoencoder = AutoEncoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_func = nn.MSELoss()
loss_train = np.zeros((epochs, 1))

# 训练
for epoch in range(epochs):
    # 不需要label，所以用一个占位符"_"代替
    for batchidx, (x, _) in enumerate(trainDataLoader):
        # 编码和解码
        encoded, decoded = autoencoder(x)
        # 计算loss
        loss = loss_func(decoded, x)
        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train[epoch, 0] = loss.item()
    print('Epoch: %04d, Training loss=%.8f' %
          (epoch + 1, loss.item()))

# 绘制loss曲线
fig = plt.figure(figsize=(6, 3))
ax = plt.subplot(1, 1, 1)
ax.grid()
ax.plot(loss_train, color=[245 / 255, 124 / 255, 0 / 255], linestyle='-', linewidth=2)
ax.set_xlabel('Epoches')
ax.set_ylabel('Loss')
plt.show()


# 重构数据
res, decodedTestdata = autoencoder(testData)
print(res)
res = res.data.numpy() #tensor - numpy
df=pd.DataFrame(res) # numpy - df
df.to_csv("transform_tem.csv",header=False,index=False,encoding="utf-8")

