import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import torch.nn as nn
import torch
from torch.autograd import *
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


df=pd.read_csv("data_02.csv")

train=df[df.columns[:3]]
y_train=df[df.columns[3:]]
# train.shape,y_train.shape

from sklearn.model_selection import train_test_split
train_x,test_x, train_y,test_y = train_test_split(train.values.reshape(-1,3), y_train.values, test_size=0.2, random_state=42)



class FA:
    def __init__(self, D, N, Beta0, gama, alpha, T, bound):
        self.D = D  # 问题维数
        self.N = N  # 群体大小
        self.Beta0 = Beta0  # 最大吸引度
        self.gama = gama  # 光吸收系数
        self.alpha = alpha  # 步长因子
        self.T = T
        self.X = (bound[1] - bound[0]) * np.random.random([N, D]) + bound[0]
        self.X_origin = copy.deepcopy(self.X)
        self.FitnessValue = np.zeros(N)
        for n in range(N):
            self.FitnessValue[n] = self.FitnessFunction(n)

    def DistanceBetweenIJ(self, i, j):
        return np.linalg.norm(self.X[i, :] - self.X[j, :])

    def BetaIJ(self, i, j):  # AttractionBetweenIJ
        return self.Beta0 * \
               np.math.exp(-self.gama * (self.DistanceBetweenIJ(i, j) ** 2))

    def update(self, i, j):
        self.X[i, :] = self.X[i, :] + \
                       self.BetaIJ(i, j) * (self.X[j, :] - self.X[i, :]) + \
                       self.alpha * (np.random.rand(self.D) - 0.5)

    def FitnessFunction(self, i):
        x_ = self.X[i, :]
        return msel[-1]

    def iterate(self):
        t = 0
        while t < self.T:
            for i in range(self.N):
                FFi = self.FitnessValue[i]
                for j in range(self.N):
                    FFj = self.FitnessValue[j]
                    if FFj < FFi:
                        self.update(i, j)
                        self.FitnessValue[i] = self.FitnessFunction(i)
                        FFi = self.FitnessValue[i]
            t += 1

    def find_min(self):
        v = np.min(self.FitnessValue)
        n = np.argmin(self.FitnessValue)
        return v, self.X[n, :]

class test_cnn(nn.Module):
    def __init__(self):
        super(test_cnn, self).__init__()
        # self.conv1 = nn.Conv1d(1, 16, 50, padding=10)
        # self.conv2 = nn.Conv1d(16, 32, 5, padding=1)
        # self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(3, FA.X_origin[:, 0])
        self.fc2 = nn.Linear(FA.X_origin[:, 0], FA.X_origin[:, 1])
        self.fc3 = nn.Linear(FA.X_origin[:, 1], 2)

    def forward(self, x):
        # x = F.max_pool1d(F.relu(self.conv1(x)), 10)
        # x = F.avg_pool1d(F.relu(self.conv1(x)),10)
        # x = F.max_pool1d(F.relu(self.conv2(x)), 7)
        # x = F.max_pool1d(F.relu(self.conv3(x)), 3)
        # x = x.view(x.size()[0], -1)  # 展开成一维的
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)



t = np.zeros(10)
value = np.zeros(10)
for i in range(10):
    fa = FA(2, 20, 1, 0.000001, 0.97, 100, [0, 100])
    time_start = time.time()
    fa.iterate()
    time_end = time.time()
    t[i] = time_end - time_start
    value[i], n = fa.find_min()
print("平均值：", np.average(value))
print("最优值：", np.min(value))
print("最差值：", np.max(value))
print("平均时间：", np.average(t))







