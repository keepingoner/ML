# -*- encoding:utf-8 -*-

"""
使用k-近邻算法预测sklearn 鸢尾花
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# 导入数据
iris = load_iris()

iris_x = iris.data

iris_y = iris.target

# 数据分为测试集和训练集
iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(iris_x, iris_y, test_size=0.25)

# 特征工程 标准化

std = StandardScaler()

iris_x_train = std.fit_transform(iris_x_train)

iris_x_test = std.transform(iris_x_test)

# 1


# knn = KNeighborsClassifier(n_neighbors=4)

# knn.fit(iris_x_train, iris_y_train)
#
# iris_y_predict = knn.predict(iris_x_test)
#
# # # 调用该对象的测试方法，主要接收一个参数：测试数据集
# probility = knn.predict_proba(iris_x_test)
#
# # print(probility)
#
# print("预测准确率为： ", knn.score(iris_x_test, iris_y_test))
#


# 2


# 算法调优 网格搜索 交叉验证


knn = KNeighborsClassifier()

param = {"n_neighbors": [3, 5, 7]}

gc = GridSearchCV(knn, param_grid=param, cv=3)

gc.fit(iris_x_train, iris_y_train)

# 预测准确率

print(gc.score(iris_x_test, iris_y_test))

# 交叉验证中最好的结果

print(gc.best_score_)

# 最好的模型

print(gc.best_estimator_)

# 每个k的 验证结果

print(gc.cv_results_)
