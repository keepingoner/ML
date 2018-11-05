# -*- encoding:utf-8 -*-

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
data = pd.read_csv("ha.csv", encoding="utf8")
print(data)
X = data.iloc[0: -1, 0:1]
# print(X)
Y = data.iloc[1:, 1:2]
# print(Y)

# 编码
# one_hot_encoder = OneHotEncoder(sparse=False, n_values=36)
# X = one_hot_encoder.fit_transform(X)
# Y = one_hot_encoder.transform(Y)


# 测试集 训练集划分

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# 建模
# lr = LinearRegression()
lr = Ridge()
lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)
score = lr.score(x_test, y_test)



print('预测值{}'.format(y_predict))

print('真是值{}'.format(y_test))

print("lr分数{}".format(score))

"""
岭回归
"""

# ridge = Ridge(alpha=0.5)
#
# ridge.fit(x_train, y_train)
#
# y_predict2 = ridge.predict(x_test)
#
# score = ridge.score(x_test, y_test)

# print("ridge分数{}".format(score))
t = np.arange(len(x_test))


pre = pd.DataFrame(data=y_predict, columns=["a"])
plt.figure()
plt.plot(t, pre["a"], 'r-', linewidth=2, label='Predict_b')
plt.plot(t, y_test, 'g-', linewidth=2, label='test')
plt.legend(loc='upper right')

plt.show()