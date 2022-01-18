import torch.nn as nn
import torch
from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import time

class BiLSTMNet(nn.Module):

    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x.view(len(x), 1, -1))  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        #         print(out.shape)
        return out

def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

import joblib
if __name__ == "__main__":
    net=joblib.load(filename="bilstm02_model.joblib")



def pre(x):
    x=x.reshape(-1,1,3)
    var_x = ToVariable(x)
    out = net(var_x)
    return out.detach().numpy()

def Bounds(s,Lb,Ub):
    temp = s
    for i in range(len(s)):
        if temp[i]<=Lb[i]:
            temp[i]=Lb[i]
        elif temp[i]>Ub[i]:
            temp[i]=Ub[i]
    return temp

def fun(X):
    test=np.array([X[0],X[1],X[2]])
    out1=pre(test)
    #out2=np.sum(np.square(test))
    
    if out1[0][0]>=0.4576:
        out2= 0
    elif out1[0][0]<=0.356:
        out2 = 0
    else :
        out2=-out1[0][1]
    return out2

''' Chebyshev映射 '''

import math

def Chebyshev(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        a = 5.78
        x[i+1] = math.cos(a*math.acos(x[i]))
    return x

''' Singer映射 '''

def Singer(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        u = 1.07  # u /in [0.9,1.08]
        x[i+1] = u * (7.86 * x[i] - 23.31 * (x[i] ** 2) + 28.75 * (x[i] ** 3) - 13.302875 *( x[i] ** 4))
    return x   # x /in [0,1]


''' Sine映射 '''

import math

def Sine(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        a=3  # a /in (0,4]
        x[i + 1] = 4/a * math.sin(math.pi*x[i])

    return x  # x /in [-1,1]

''' Cubic映射 '''
def Cubic(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        a = 4
        b = 3
        x[i+1] = a * x[i]**3 - b * x[i]

    return x  # x /in [-1,1]

''' Tent映射 '''

def Tent(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()  # 初始点
    a = 0.7  # 参数a的值
    for i in range(Max_iter - 1):
        if x[i] < a:
            x[i + 1] = x[i] / a
        if x[i] >= a:
            x[i + 1] = (1 - x[i]) / (1 - a)

    return x


''' Logistic映射 '''
def Logistic(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    if x[0] == 0. :
        x[0] = random.random()
    elif x[0] == 0.25 :
        x[0] = random.random()
    elif x[0] == 0.5 :
        x[0] = random.random()
    elif x[0] == 0.75 :
        x[0] = random.random()
    else :
        x[0] = x[0]
    u = 4    # 这个4很重要 由logistic混沌映射决定的,具体可以看logistic的性质
    for i in range(Max_iter - 1):
        x[i+1] = u * x[i] * (1-x[i])
    return x


# pop是种群，M是迭代次数，f是用来计算适应度的函数
# pNum是生产者
def SSA(pop,M,cc,dd,dim):
    #net=joblib.load("bilstm01_net.joblib")
    #global fit
    P_percent=0.2
    pNum = round(pop*P_percent)  # 生产者的人口规模占总人口规模的20%
    #lb = c*np.ones((1,dim))  # 生成1*dim的全1矩阵，并全乘以c；lb是下限
    #ub = d*np.ones((1,dim))  # ub是上限
    lb=cc
    ub=dd
    X = np.zeros((pop,dim))  # 生成pop*dim的全0矩阵，代表麻雀位置
    fit = np.zeros((pop,1))   # 适应度值初始化

    # for i in range(pop):
    #     X[i,:] = lb+(ub-lb)*np.random.rand(1,dim)  # 麻雀属性随机初始化初始值
    #
    # for i in range(pop):
    #     for j in range(dim):
    #             X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    for j in range(dim):
        # ChebyValue = Chebyshev(pop)
        # TentValue = Tent(pop)
        # LogValue = Logistic(pop)
        # SingerValue = Singer(pop)
        # SineValue = Sine(pop)
        CubicValue = Cubic(pop)
        for i in range(pop):
            # X[i, j] = ChebyValue[i] * ((ub[j] - lb[j])/2) + ((ub[j] - lb[j])/2)
            # X[i, j] = TentValue[i] * (ub[j] - lb[j]) + lb[j]
            # X[i, j] = LogValue[i] * (ub[j] - lb[j]) + lb[j]
            # X[i, j] = SingerValue[i] * (ub[j] - lb[j]) + lb[j]
            # X[i, j] = SineValue[i] * ((ub[j] - lb[j]) / 2) + ((ub[j] - lb[j]) / 2)
            X[i, j] = CubicValue[i] * ((ub[j] - lb[j]) / 2) + ((ub[j] - lb[j]) / 2)
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]

            fit[i,0] = fun(X[i,:])  # 初始化最佳适应度值


    pFit = fit  #最佳适应度矩阵
    pX = X  # 最佳种群位置
    fMin = np.min(fit[:,0]) # fMin表示全局最优适应值，生产者能量储备水平取决于对个人适应度值的评估
    bestI = np.argmin(fit[:,0])
    bestX1 = X[bestI,:].copy() # bestX表示fMin对应的全局最优位置的变量信息
    Convergence_curve = np.zeros((1,M))  # 初始化收敛曲线
    for t in range(M): # 迭代更新
    
        sortIndex = np.argsort(pFit.T)  # 对麻雀的适应度值进行排序，并取出下标
        fmax = np.max(pFit[:,0])  # 取出最大的适应度值
        B = np.argmax(pFit[:,0])  # 取出最大的适应度值得下标
        worse = X[B,:]  # 最差适应度

        r2 = np.random.rand(1) # 预警值
        # 这一部位为发现者（探索者）的位置更新
        if r2 < 0.8: # 预警值较小，说明没有捕食者出现
            for i in range(pNum):
                r1 = np.random.rand(1)
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]*np.exp(-(i)/(r1*M))  # 对自变量做一个随机变换
                X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub)  # 对超过边界的变量进行去除
                fit[sortIndex[0,i],0] = fun(X[sortIndex[0,i],:])   # 算新的适应度值
        elif r2 >= 0.8: # 预警值较大，说明有捕食者出现威胁到了种群的安全，需要去其它地方觅食
            for i in range(pNum):
                Q = np.random.rand(1,dim)  # 也可以替换成  np.random.normal(loc=0, scale=1.0, size=1)
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]+Q*np.ones((1,dim))  # Q是服从正态分布的随机数。L表示一个1×d的矩阵
                X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub)
                fit[sortIndex[0,i],0] = fun(X[sortIndex[0,i],:])
        bestII = np.argmin(fit[:,0])
        bestXX = X[bestII,:].copy()
        #  这一部位为加入者（追随者）的位置更新
        for ii in range(pop-pNum):
            #print(bestX1)
            i = ii+pNum
            A = np.floor(np.random.rand(1,dim)*2)*2-1
            if i > pop/2:  #  这个代表这部分麻雀处于十分饥饿的状态（因为它们的能量很低，也就是适应度值很差），需要到其它地方觅食
                Q = np.random.rand(1,dim)  # 也可以替换成  np.random.normal(loc=0, scale=1.0, size=1)
                X[sortIndex[0,i],:] = Q*np.exp(worse-pX[sortIndex[0,i],:]/np.square(i))
            else:  # 这一部分追随者是围绕最好的发现者周围进行觅食，其间也有可能发生食物的争夺，使其自己变成生产者
                X[sortIndex[0,i],:] = bestXX+np.dot(np.abs(pX[sortIndex[0,i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1,dim))
            X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub)
            fit[sortIndex[0,i],0] = fun(X[sortIndex[0,i],:])

        # 这一部位为意识到危险（注意这里只是意识到了危险，不代表出现了真正的捕食者）的麻雀的位置更新
        # np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是5，步长为1。
        # 一个参数时，参数值为终点，起点取默认值0，步长取默认值1
        arrc = np.arange(len(sortIndex[0,:]))
        #c=np.random.shuffle(arrc)
        # 这个的作用是在种群中随机产生其位置（也就是这部分的麻雀位置一开始是随机的，意识到危险了要进行位置移动，
        #  处于种群外围的麻雀向安全区域靠拢，处在种群中心的麻雀则随机行走以靠近别的麻雀）
        c = np.random.permutation(arrc)  # 随机排列序列
        b = sortIndex[0,c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0,b[j]],0] > fMin:
                X[sortIndex[0,b[j]],:] = bestX1+np.random.rand(1,dim)*np.abs(pX[sortIndex[0,b[j]],:]-bestX1)
            else:
                X[sortIndex[0,b[j]],:] = pX[sortIndex[0,b[j]],:]+(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0,b[j]],:]-worse)/(pFit[sortIndex[0,b[j]]]-fmax+10**(-50))
            X[sortIndex[0,b[j]],:] = Bounds(X[sortIndex[0,b[j]],:],lb,ub)
            fit[sortIndex[0,b[j]],0] = fun(X[sortIndex[0,b[j]],:])
        print(bestX1)
        #print(fMin)
        for i in range(pop):
            #fitsev=fun(X[i,:])

            if fit[i,0] < pFit[i,0]:
                pFit[i,0] = fit[i,0]
                pX[i,:] = X[i,:]
            if pFit[i,0] < fMin:
                fMin = pFit[i,0]
                bestX1 = pX[i,:].copy()
                
        Convergence_curve[0,t] = fMin
        print(fMin)
        print(bestX1)
        print(fun(bestX1))
        print(pre(bestX1))

        if fMin<-0.6417:
            dptime.append(time.process_time())


    return fMin,bestX1,Convergence_curve


pop=100
M=150
cc=np.array([0,0,0])
dd=np.array([1.,1.,1.])
dptime=[]
dim=3

[fMin,bestX,SSA_curve]=SSA(pop,M,cc,dd,dim)
fMin=-fMin

SSA_curve=-SSA_curve
print(['最优值为：',fMin])
print(['最优变量为：',bestX])
time = time.process_time()
print('Running time: %s Seconds' % (time ))
thr1=np.arange(len(SSA_curve[0,:]))
print('达到目标的最小迭代时间:',dptime[0])

plt.plot(thr1, SSA_curve[0,:])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('iteration',fontsize=15)
plt.ylabel('object value',fontsize=15)
plt.title('Cubic',fontsize=15)
plt.show()

