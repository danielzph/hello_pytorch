from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import numpy as np
import pandas as pd

# # 加载数据集
# boston_data = load_boston()
# print(boston_data)
#
# # 拆分数据集
# x = boston_data.data
# y = boston_data.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# 加载数据集
df=pd.read_csv("data_02.csv")

x=df[df.columns[:3]]
y=df[df.columns[3:]]

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x.values.reshape(-1,3), y.values, test_size=0.2, random_state=42)


# 预处理
y_train = np.array(y_train).reshape(-1, 2)
y_test = np.array(y_test).reshape(-1, 2)
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
y_train = StandardScaler().fit_transform(y_train)
y_test = StandardScaler().fit_transform(y_test)


#创建svR实例
svr=SVR(C=1, kernel='rbf', epsilon=0.2)
svr=svr.fit(x_train,y_train)
#预测
svr_predict=svr.predict(x_test)
#评价结果
mae = mean_absolute_error(y_test, svr_predict)
mse = mean_squared_error(y_test, svr_predict)
evs = explained_variance_score(y_test, svr_predict)
r2 = r2_score(y_test, svr_predict)
print("MAE：", mae)
print("MSE：", mse)
print("EVS：", evs)
print("R2：", r2)


