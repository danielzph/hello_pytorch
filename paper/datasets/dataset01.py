import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

def get_files(root1,root2):
    data_root1=root1
    data_root2=root2
    train_df = pd.read_csv(data_root1)
    y_train_df = pd.read_csv(data_root2)
    train = np.array(train_df)
    y_train = np.array(y_train_df)
    return train, y_train

# 实例化
train, y_train = get_files('../data/data_x','../data/data_y')



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



# 构造Dataset对象
dataset = MyDataset(train,y_train)
# 构造DataLoader对象
dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=False)


