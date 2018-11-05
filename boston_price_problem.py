# -*- encoding:utf-8 -*-
"""
波士顿房价
"""
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 获取数据

data = load_boston()

# 分割数据 测试集  训练集


x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

# 标准化处理 特征值和目标值都处理  实例化两个标准化   结果用inverse_transform() 返回标准化之前的值

std_x = StandardScaler()

x_train = std_x.fit_transform(x_train)

x_test = std_x.transform(x_test)

std_y = StandardScaler()

# 0。19  0。18版本不同   需要reshape转化为2维形状
y_train = std_y.fit_transform(y_train.reshape(-1, 1))

y_test = std_y.transform(y_test.reshape(-1, 1))

# 正规方程求解

lr = LinearRegression()

lr.fit(x_train, y_train)

print(lr.coef_)

y_predict = lr.predict(x_test)

# 返回标准化之前的值
y_predict = std_y.inverse_transform(y_predict)

print("预测价格", y_predict)


print("正规方程的均方误差", mean_squared_error(std_y.inverse_transform(y_test), std_y.inverse_transform(y_predict)))

# 通过梯度下降法

sgd = SGDRegressor()

sgd.fit(x_train, y_train)

print(sgd.coef_)

y_predict = sgd.predict(x_test)

y_predict = std_y.inverse_transform(y_predict)

print("预测价格", y_predict)

