import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_files(root, test=False):
    data_root1='../example_use_pytorch/CNN_test/data2'
    data_root2='../example_use_pytorch/CNN_test/label'








# 自定义Dataset的子类
class MyDataset(Dataset):
    # 构造器初始化方法
    def __init__(self, data_X, data_y):
        self.data_X = data_X
        self.data_y = data_y

    # 重写getitem方法用于通过idx获取数据内容
    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]

    # 重写len方法获取数据集大小
    def __len__(self):
        return self.data_X.shape[0]



# # 构造Dataset对象
# dataset = MyDataset(data_X, data_y)
# # 构造DataLoader对象
# dataloader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True, drop_last=False)
