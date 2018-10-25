# -*- encoding:utf-8 -*-

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv("ha.csv", encoding="utf8")
X = data.iloc[0: -1, 1:]
Y = data.iloc[1:, 1:]

# 数据归一化
# stand_scaler = StandardScaler()
# X = stand_scaler.fit_transform(X)
# Y = stand_scaler.transform(Y)
# 测试集 训练集划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# 建模

lr = LinearRegression()
# lr = Ridge()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
# y_predict = stand_scaler.inverse_transform(y_predict)
score = lr.score(x_test, y_test)

# print('预测值{}'.format(y_predict))
# print('真实值{}'.format(y_test))
print("lr分数{}".format(score))
print(lr.get_params())
t = np.arange(len(y_test))

da = pd.DataFrame(data=y_predict, dtype=np.int, columns=["a",
                                                         "b",
                                                         "c",
                                                         "d",
                                                         "e",
                                                         "f",
                                                         "g"])
# print(da)
# y_test = stand_scaler.inverse_transform(y_test)
# print(y_test)
plt.subplot(2, 4, 1)
plt.plot(t, y_test["a"], 'r-')
plt.plot(t, da["a"], 'g-')

plt.subplot(2, 4, 2)
plt.plot(t, y_test["b"], 'r-', linewidth=2, label='Test_b')
plt.plot(t, da["b"], 'g-', linewidth=2, label='Predict_b')

plt.subplot(2, 4, 3)
plt.plot(t, y_test["c"], 'r-', linewidth=2, label='Test_c')
plt.plot(t, da["c"], 'g-', linewidth=2, label='Predict_c')

plt.subplot(2, 4, 4)
plt.plot(t, y_test["d"], 'r-', linewidth=2, label='Test_d')
plt.plot(t, da["d"], 'g-', linewidth=2, label='Predict_d')
#
plt.subplot(2, 4, 5)
plt.plot(t, y_test["e"], 'r-', linewidth=2, label='Test_e')
plt.plot(t, da["e"], 'g-', linewidth=2, label='Predict_e')

plt.subplot(2, 4, 7)
plt.plot(t, y_test["f"], 'r-', linewidth=2, label='Test_f')
plt.plot(t, da["f"], 'g-', linewidth=2, label='Predict_f')

plt.subplot(2, 4, 8)
plt.plot(t, y_test["g"], 'r-', linewidth=2)
plt.plot(t, da["g"], 'g-', linewidth=2)

plt.legend(loc='upper right')

plt.grid()
plt.show()
