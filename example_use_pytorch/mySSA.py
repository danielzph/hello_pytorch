import torch.nn as nn
import torch
from torch.autograd import *
import numpy as np
import random
from matplotlib import pyplot as plt
import torch.nn.functional as F
# from bp11 import test_cnn


# from mpl_toolkits.mplot3d import Axes3D

# class BiLSTMNet(nn.Module):
#
#     def __init__(self, input_size):
#         super(BiLSTMNet, self).__init__()
#         self.rnn = nn.LSTM(
#             input_size=input_size,
#             hidden_size=32,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.out = nn.Sequential(
#             nn.Linear(64, 2)
#         )
#
#     def forward(self, x):
#         r_out, (h_n, h_c) = self.rnn(x.view(len(x), 1, -1))  # None 表示 hidden state 会用全0的 state
#         out = self.out(r_out[:, -1])
#         #         print(out.shape)
#         return out

class test_cnn(nn.Module):
    def __init__(self):
        super(test_cnn, self).__init__()
        # self.conv1 = nn.Conv1d(1, 16, 50, padding=10)
        # self.conv2 = nn.Conv1d(16, 32, 5, padding=1)
        # self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(3, 80)
        self.fc2 = nn.Linear(80, 20)
        self.fc3 = nn.Linear(20, 2)

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


import joblib
if __name__ == "__main__":
    net=joblib.load(filename="bp_model.joblib")



def pre(x):
    x=x.reshape(-1,3)
    var_x = ToVariable(x)
    out = net(var_x)
    return out.detach().numpy()


'''优化函数'''


def fun(X):
    out1=pre(X)
    if out1[0, 0] <= 0.245:
        out2=-0
    elif out1[0, 0] >= 0.265:
        out2=-0
    else:
        out2 = -out1[0, 1]

    return out2


''' 种群初始化函数 '''

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X

''' Chebyshev映射 '''
#
# import math
#
# def Chebyshev(Max_iter):
#     x = np.zeros([Max_iter, 1])
#     x[0] = random.random()
#     for i in range(Max_iter - 1):
#         a = 5.78
#         x[i+1] = math.cos(a*math.acos(x[i]))
#     return x
#
#
#
''' 种群初始化函数--改进版本 '''
# def ipinitial(pop, dim, ub, lb):
#     X = np.zeros([pop, dim])
#     for j in range(dim):
#         ChebyValue = Chebyshev(pop)
#         # TentValue = Tent(pop)
#         # LogValue = Logistic(pop)
#         for i in range(pop):
#             X[i, j] = ChebyValue[i] * ((ub[j] - lb[j])/2) + ((ub[j] - lb[j])/2)
#             # X[i, j] = TentValue[j] * (ub[j] - lb[j]) + lb[j]
#             # X[i, j] = LogValue[j] * (ub[j] - lb[j]) + lb[j]
#             if X[i, j] > ub[j]:
#                 X[i, j] = ub[j]
#             if X[i, j] < lb[j]:
#                 X[i, j] = lb[j]
#
#     return X

'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''麻雀发现者更新'''


def PDUpdate(X, PDNumber, ST, Max_iter):
    X_new = X
    R2 = random.random()
    for j in range(PDNumber):
        if R2 < ST:
            X_new[j, :] = X[j, :] * np.exp(-j / (random.random() * Max_iter))
        else:
            X_new[j, :] = X[j, :] + np.random.randn() * np.ones([1, dim])
    return X_new


'''麻雀加入者更新'''


def JDUpdate(X, PDNumber, pop, dim):
    X_new = X
    for j in range(PDNumber + 1, pop):
        if j > (pop - PDNumber) / 2 + PDNumber:
            X_new[j, :] = np.random.randn() * np.exp((X[-1, :] - X[j, :]) / j ** 2)
        else:
            # 产生-1，1的随机数
            A = np.ones([dim, 1])
            for a in range(dim):
                if (random.random() > 0.5):
                    A[a] = -1
        AA = np.dot(A, np.linalg.inv(np.dot(A.T, A)))
        X_new[j, :] = X[1, :] + np.abs(X[j, :] - X[1, :]) * AA.T

    return X_new


'''危险更新'''


def SDUpdate(X, pop, SDNumber, fitness, BestF):
    X_new = X
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[0:SDNumber]
    for j in range(SDNumber):
        if fitness[SDchooseIndex[j]] > BestF:
            X_new[SDchooseIndex[j], :] = X[1, :] + np.random.randn() * np.abs(X[SDchooseIndex[j], :] - X[1, :])
        elif fitness[SDchooseIndex[j]] == BestF:
            K = 2 * random.random() - 1
            X_new[SDchooseIndex[j], :] = X[SDchooseIndex[j], :] + K * (
                        np.abs(X[SDchooseIndex[j], :] - X[-1, :]) / (fitness[SDchooseIndex[j]] - fitness[-1] + 10E-8))
    return X_new


'''麻雀搜索算法'''


def SSA(pop, dim, lb, ub, Max_iter, fun):
    ST = 0.8  # 预警值
    PD = 0.2  # 发现者的比列，剩下的是加入者
    SD = 0.2  # 意识到有危险麻雀的比重
    PDNumber = int(pop * PD)  # 发现者数量
    SDNumber = int(pop - pop * PD)  # 意识到有危险麻雀数量
    X = initial(pop, dim, ub, lb)  # 初始化种群
    # X = ipinitial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = X[0, :]
    Curve = np.zeros([MaxIter, 1])
    for i in range(MaxIter):

        BestF = fitness[0]

        X = PDUpdate(X, PDNumber, ST, Max_iter)  # 发现者更新

        X = JDUpdate(X, PDNumber, pop, dim)  # 加入者更新

        X = SDUpdate(X, pop, SDNumber, fitness, BestF)  # 危险更新

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon = X[0, :]
        Curve[i] = GbestScore
        print('迭代次数：', i)
        print('该次最优适应度值：',GbestScore)
        print('该次最优位置：',GbestPositon)

    return GbestScore, GbestPositon, Curve


'''主函数 '''
# 设置参数
pop =100 # 种群数量
MaxIter = 1000  # 最大迭代次数
dim = 3  # 维度
lb = 0 * np.ones([dim, 1])  # 下边界
ub = 1 * np.ones([dim, 1])  # 上边界

GbestScore, GbestPositon, Curve = SSA(pop, dim, lb, ub, MaxIter, fun)  # 纵横交叉算法
print('最优适应度值：', GbestScore)
print('最优解：', GbestPositon)

Curve=-Curve
# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('SSA', fontsize='large')
plt.show()

