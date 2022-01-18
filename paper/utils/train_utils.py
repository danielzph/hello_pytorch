import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
# import models
from ..models.CNN_1d import CNN as CNN_1d
from ..models.CNN_2d import CNN as CNN_2d
from ..models.BiLSTM1d import BiLSTM as BiLSTM_1d
from ..models.BiLSTM2d import BiLSTM as BiLSTM_2d
from torch.autograd import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



# ToVariable
def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

# 模型 误差 优化器
net = BiLSTM_1d               # 暂时需要改
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)


# 训练模式
class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir


    def setup(self):
        args = self.args
        # gpu or cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))



# train_x train_y
    def train(self,epoch,batchsize):
        """
        Training process
        :return:
        """
        msel = []
        mael = []
        lossl = []
        vallossl = []
        for e in range(epoch):
            avgloss = 0
            for i in range(batchsize, len(train_x), batchsize):
                var_x = ToVariable(train_x[i - batchsize:i])
                var_y = ToVariable(train_y[i - batchsize:i])
                # forwardss
                out = net(var_x)
                loss = criterion(out, var_y)
                # backward
                optimizer.zero_grad()
                loss.backward()
                avgloss += loss.data / len(train_x)
                optimizer.step()
            lossl.append(avgloss)
            val_avgloss = 0
            for i in range(batchsize, len(val_x), batchsize):
                var_x = ToVariable(val_x[i - batchsize:i])
                var_y = ToVariable(val_y[i - batchsize:i])
                out = net(var_x)
                loss = criterion(out, var_y)
                val_avgloss += loss.data / len(val_x)
            vallossl.append(val_avgloss)
            print("epoch :{} train_loss :{}  val_loss:{}".format(e, avgloss, val_avgloss))

            var_x = ToVariable(val_x)
            var_y = ToVariable(val_y)
            out = net(var_x)
            mse = mean_squared_error(var_y, out.detach().numpy())
            mae = mean_absolute_error(var_y, out.detach().numpy())
            msel.append(mse)
            mael.append(mae)
            print("val mse:{} val mae:{}".format(mse, mae))
