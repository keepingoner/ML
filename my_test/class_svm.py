# -*- encoding:utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC

data = pd.read_table('double.txt', header=None, encoding='gb2312', delim_whitespace=True, index_col=0)
print(data.head())
#
X = data.iloc[0: -1, 1:7]

Y = data.iloc[1:, 7:8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# 建模
# lr = LinearRegression()
# lr = Ridge()
lr = SVC(kernel="linear")
# lr = LinearSVC()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
score = lr.score(x_test, y_test)



# print('预测值{}'.format(y_predict))

# print('真是值{}'.format(y_test))

print("lr分数{}".format(score))

# for neighbors in range(1, 30):
#
#     lr = KNeighborsClassifier(n_neighbors=neighbors)
#     lr.fit(x_train, y_train)
#
#     y_predict = lr.predict(x_test)
#
#     score = lr.score(x_test, y_test)
#
#     # 作图
#     import numpy as np
#
#     t = np.arange(len(x_test))
#     print(type(t), type(y_predict))
#     plt.figure()
#     # plt.scatter(t[:50], y_predict[:50], s=10, c="b")
#     # plt.scatter(t[:50], y_test[:50], s=10, c="r")
#
#     plt.plot(t[:50], y_predict[:50], 'c--', linewidth=2, label='Test_d')
#     plt.plot(t[:50], y_test[:50], 'm-', linewidth=2, label='Predict_d')
#
#
#     plt.xticks(tuple([x for x in range(len(x_test[:50]))]))
#
#     plt.grid()  # 生成网格
#
#     plt.show()
#
#     # print(y_predict)
#     print(score)

# 交叉验证 选取最好 n
# knn = KNeighborsClassifier()
#
# param = {"n_neighbors": [x for x in range(1, 20)]}

# gc = GridSearchCV(knn, param_grid=param, cv=3)
#
# gc.fit(x_train, y_train)
#
# # 预测准确率
#
# print(gc.score(x_test, y_test))
#
# # 交叉验证中最好的结果
#
# # print(gc.best_score_)
#
# # 最好的模型
#
# print(gc.best_estimator_)

# 每个k的 验证结果

# print(gc.cv_results_)


