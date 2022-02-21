import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# df=pd.read_csv("../data/shiyan/tem02.csv")
df=pd.read_csv("CNN_test/f_data06.csv")
train=df[df.columns[13:14]]
# train.values.reshape(-1,1)
# print(train)
train=np.array(train)
train=train[:4356]
# print(train)
j=0
tem=[]
newtrain=[0]*198   # 7数出来的 后面要改
for i in range(len(train)):
    tem.append(train[i][0])
    if (i+1)%22==0:
        newtrain[j]=tem
        j = j + 1
        tem=[]
# print(newtrain)

train_x,test_x= train_test_split(newtrain, test_size=1/9, random_state=2)
# print(train_x)
# print(test_x)



#线性自编码器
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(22, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            nn.Linear(8, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            # nn.Linear(32, 64),
            # nn.Tanh(),
            nn.Linear(16, 22),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#参数设定
epochs = 2000
BATCH_SIZE = 10
LR = 0.0001

# 数据类型转换
trainData = torch.FloatTensor(train_x)
testData = torch.FloatTensor(test_x)
newtrain=torch.FloatTensor(newtrain)



# 构建张量数据集
train_dataset = TensorDataset(trainData, trainData)
test_dataset = TensorDataset(testData, testData)
trainDataLoader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

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

# 利用训练好的自编码器重构测试数据
res, decodedTestdata = autoencoder(newtrain)
# print(res)
# print(res.shape)
res=res.detach().numpy().tolist()
# print(res)
out=[]
for n in range(0,len(res)):
    for m in range(0,len(res[0])):
        out.append(res[n][m])
print(out)
Out=[[0] for _ in range(len(out))]
for l in range(0,len(out)):
    Out[l][0]=out[l]

df=pd.DataFrame(Out)
df.to_excel("../data/shiyan/ae_z_i_14.xlsx",header=True,index=False,encoding="utf-8")








# # 绘图
# fig = plt.figure(figsize=(6, 9))
# for i in range(len(test_x)):
#     ax = plt.subplot(3, 1, i + 1)
#     ax.plot(test_x[:i], color='C0', linestyle='-', linewidth=2)
#     # ax.plot(reconstructedData[:, i], color='C3', linestyle='-', linewidth=2)
#     ax.legend(['Real value', 'Reconstructed value'], loc="upper right",
#               edgecolor='black', fancybox=True)
# plt.show()
#
# print(autoencoder.encoder(test_x))











# starttime = time.time()
# torch.manual_seed(1)
# EPOCH = 10
# BATCH_SIZE = 64
# LR = 0.005
# N_TEST_IMG = 5
#
# train_data = torchvision.datasets.MNIST(
#     root='MINIST',
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=True
# )
# loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# # print(train_data.train_data[0])
# plt.imshow(train_data.train_data[2].numpy(),cmap='Greys')
# plt.title('%i'%train_data.train_labels[2])
# plt.show()


# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 128),
#             nn.Tanh(),
#             nn.Linear(128, 64),
#             nn.Tanh(),
#             nn.Linear(64, 32),
#             nn.Tanh(),
#             nn.Linear(32, 16),
#             nn.Tanh(),
#             nn.Linear(16, 3)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(3, 16),
#             nn.Tanh(),
#             nn.Linear(16, 32),
#             nn.Tanh(),
#             nn.Linear(32, 64),
#             nn.Tanh(),
#             nn.Linear(64, 128),
#             nn.Tanh(),
#             nn.Linear(128, 28 * 28),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded


'''
Coder = AutoEncoder()
print(Coder)

optimizer = torch.optim.Adam(Coder.parameters(),lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(loader):
        b_x = x.view(-1,28*28)
        b_y = x.view(-1,28*28)
        b_label = y
        encoded , decoded = Coder(b_x)
        loss = loss_func(decoded,b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5 == 0:
            print('Epoch :', epoch,'|','train_loss:%.4f'%loss.data)

torch.save(Coder,'AutoEncoder.pkl')
print('________________________________________')
print('finish training')

endtime = time.time()
print('训练耗时：',(endtime - starttime))
'''
#
# Coder = AutoEncoder()
# Coder = torch.load('AutoEncoder.pkl')

'''
#数据的空间形式的表示
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = Coder(view_data)    # 提取压缩的特征值
fig = plt.figure(2)
ax = Axes3D(fig)    # 3D 图
# x, y, z 的数据值
X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()
# print(X[0],Y[0],Z[0])
values = train_data.train_labels[:200].numpy()  # 标签值
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))    # 上色
    ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
'''

# 原数据和生成数据的比较
# plt.ion()
# plt.show()

# for i in range(10):
#     test_data = train_data.train_data[i].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
#     _, result = Coder(test_data)

    # print('输入的数据的维度', train_data.train_data[i].size())
    # print('输出的结果的维度',result.size())

#     im_result = result.view(28, 28)
#     # print(im_result.size())
#     plt.figure(1, figsize=(10, 3))
#     plt.subplot(121)
#     plt.title('test_data')
#     plt.imshow(train_data.train_data[i].numpy(), cmap='Greys')
#
#     plt.figure(1, figsize=(10, 4))
#     plt.subplot(122)
#     plt.title('result_data')
#     plt.imshow(im_result.detach().numpy(), cmap='Greys')
#     plt.show()
#     plt.pause(0.5)
#
# plt.ioff()

