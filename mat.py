# -*- encoding:utf-8 -*-

# @Time    : 2018/9/20 15:01
# @Author  : jin
# @File    : templet.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasets = pd.read_csv("datasets/studentscores.csv")
print(datasets.head(3))
X = datasets["Hours"]
Y = datasets["Scores"]
X2 = [1,2,3,4]
Y2 = [8,9,10,11]

# 绘图
# 设置画布大小
# plt.figure(figsize=(6, 6))
# # 放入x， y 数据  可以放入多条数据
# plt.plot(X2, Y2, "b*--", label="11111")
# plt.plot(X, Y, "r*-", label="2222")
# # 设置标题1
# plt.title("分数和小时关系图")
# # 通过 xlim 和 ylim 来设限定轴的范围，通过 xlabel 和 ylabel 来设置轴的名称。
# plt.xlim(0, 30)
# plt.ylim(0, 100)
# plt.xlabel("Hour")
# plt.ylabel("Score")
#
# # 通过 xticks 和 yticks 来设置轴的刻度。
# # plt.xticks((0,2,4))
# # 设置 label 和 legend 的目的就是为了区分出每个数据对应的图形名称。
# # plt.plot(X2, Y2, label="11111")
# # plt.plot(X, Y, label="2222")
# plt.legend(loc='best')
# # 中文乱码问题
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.show()


#散点图
# plt.figure(figsize=(6, 6))
#
# # size = 50 # 生成每个点的大小
# # colour = np.arctan2(X, Y) # 生成每个点的颜色大小
# size = 10
# colour = "r"
# plt.scatter(X, Y, s=size, c=colour)
# # plt.colorbar() # 添加颜色栏
#
# # 中文乱码问题
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.show()


# # 柱状图
#
# plt.bar(X, Y) # 画出 x 和 y 的柱状图
#
# # # 增加数值
# for x, y in zip(X, Y):
#     plt.text(x, y , '%.2f' % y, ha='center', va='bottom')
#
# plt.show()
