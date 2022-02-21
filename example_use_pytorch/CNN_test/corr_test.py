import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt

# 读数据
df=pd.read_csv("f_data06.csv")

# data_06=df[df.columns[5]]
# data_06=data_06[:4356]
# data_07=df[df.columns[6]]
# data_07=data_07[:4356]
# result1 = np.corrcoef(data_06, data_07)
# print(result1)

data_f=df[df.columns[5:10]]
data_f=data_f[:4356]
result2 = data_f.corr()
print(result2)

# 散点图
sns.pairplot(data_f)
plt.show()

#热力图
figure, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data_f.corr(), square=True, annot=True, ax=ax)
plt.show()
