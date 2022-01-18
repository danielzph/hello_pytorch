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

# 数据预处理





# cnn模型
class test_cnn(nn.Module):
    def __init__(self,num_classes=10):
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),  # 对这16个结果进行规范处理，
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # 14+2*2-5+1=14  该次卷积后output_size = 14*14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 14/2=7 池化后为7*7

        self.fc = nn.Linear(7 * 7 * 32, num_classes)
    # 前馈网络过程
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out




def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

net = test_cnn()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)





