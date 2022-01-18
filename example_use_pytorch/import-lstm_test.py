# import torch.nn as nn
# import torch
# from torch.autograd import *
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt



# import pandas as pd
# # import numpy as np
# df=pd.read_csv("make_csv.csv")
#
# train=df[df.columns[:3]]
# y_train=df[df.columns[3:]]
# train.shape,y_train.shape
#
# from sklearn.model_selection import train_test_split
# train_x,test_x, train_y,test_y = train_test_split(train.values.reshape(-1,1,3), y_train.values, test_size=0.2, random_state=42)

# Bilstm model
import torch.nn as nn
import torch
from torch.autograd import *
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np


class BiLSTMNet(nn.Module):

    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 2)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x.view(len(x), 1, -1))  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        #         print(out.shape)
        return out


def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)



# net = BiLSTMNet(test_x.shape[-1])
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)


import joblib
if __name__ == "__main__":
    net=joblib.load(filename="bilstm_model.joblib")



def pre(x):
    x=x.reshape(-1,1,3)
    var_x = ToVariable(x)
    out = net(var_x)
    return out.detach().numpy()

# print(pre(test_x))


test=np.array([0.5,1,1])

print(pre(test))


