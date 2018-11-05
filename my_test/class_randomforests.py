# -*- encoding:utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

data = pd.read_table('../happy/double.txt', header=None, encoding='gb2312', delim_whitespace=True, index_col=0)
# print(data.head())
#
X = data.iloc[0: -1, 1:7]

Y = data.iloc[1:, 7:8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

clf = RandomForestClassifier(n_estimators=30,
                             criterion="gini",
                             random_state=0,
                             max_features="sqrt",
                             max_depth=3,
                             min_samples_leaf=1)

clf.fit(X, Y)

y_predict = clf.predict(x_test)

score = clf.score(x_test, y_test)

# print('预测值{}'.format(y_predict))

# print('真是值{}'.format(y_test))

next_data = clf.predict([[03, 10, 11, 14, 15, 32]])

before_data = clf.predict([[9, 13, 14, 19, 22, 25]])

print("预测:{}".format(next_data))

print("分数{}".format(score))

# 保存
# dot -Tpng -o 1.png 1.dot
# with open('blue.dot', 'w') as f:
    # f = tree.export_graphviz(clf.get_params(), out_file=f)


# #作图
# import numpy as np
# import matplotlib.pyplot as plt
#
# t = np.arange(len(x_test))
# print(type(t), type(y_predict))
# plt.figure()
# # plt.scatter(t[:50], y_predict[:50], s=10, c="b")
# # plt.scatter(t[:50], y_test[:50], s=10, c="r")
#
# plt.plot(t[:50], y_predict[:50], 'c--', linewidth=2, label='Test_d')
# plt.plot(t[:50], y_test[:50], 'm-', linewidth=2, label='Predict_d')
#
#
# plt.xticks(tuple([x for x in range(len(x_test[:50]))]))
#
# plt.grid()  # 生成网格
#
# plt.show()
