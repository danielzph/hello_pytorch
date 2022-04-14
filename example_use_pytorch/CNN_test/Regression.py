from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score



df=pd.read_csv("f_data07.csv")
train=df[df.columns[0:14]]
y_train=df[df.columns[18]]

train=train.values.reshape(-1,22*14)
y_train=y_train.values[:198]

train_x,test_x, train_y,test_y = train_test_split(train, y_train, test_size=2/9, random_state=2)



# clf = SVR()
# clf=LinearRegression()
# clf=KNeighborsRegressor()
# clf=Ridge()
# clf=MLPRegressor()
clf=GaussianProcessRegressor()
# clf=BayesianRidge()
rf = clf.fit (train_x, train_y.ravel())
y_pred = rf.predict(test_x)
mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
EVRS= explained_variance_score(test_y,y_pred)
R2_S=r2_score(test_y,y_pred)
print("SVR结果如下：")
# print("训练集分数：",rf.score(train_x, train_y))
# print("测试集分数：",rf.score(test_x,test_y))
print("test mse:{} test mae:{} test EVRS:{} test R2_S:{}".format(mse, mae,EVRS,R2_S))






