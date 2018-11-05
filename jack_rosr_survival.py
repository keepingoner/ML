# -*- coding: utf-8 -*-

# @Time    : 2018/9/25 16:53
# @Author  : jin
# @File    : j.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
# 导入数据
data_train = pd.read_csv("data/jack_rose_train.csv")
# print(data_train.columns)
# print(data_train.info())
# print(data_train.describe())


# 画图分析
def map_plt():
    """
    画图分析数据
    :return:
    """
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
    data_train.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not.
    plt.title(u"获救情况 (1为获救)") # puts a title on our graph
    plt.ylabel(u"人数")

    plt.subplot2grid((2,3),(0,1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel(u"年龄")                         # sets the y axis lable
    plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
    plt.title(u"按年龄看获救分布 (1为获救)")


    plt.subplot2grid((2,3),(1,0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")# plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


    plt.subplot2grid((2,3),(1,2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.show()

def map2_plt():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.show()


# 缺失值处理
def set_miss():
    """
    缺失值处理
    :return:
    """
    # 使用随机森林处理年龄
    age_df = data_train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # print(age_df)
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    # print(known_age)
    # print(unknown_age)
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # print(predictedAges)

    # 用得到的预测结果填补原缺失数据
    data_train.loc[(data_train.Age.isnull()), 'Age'] = predictedAges
    # print(data_train)
    # print(data_train.columns)
    # print(data_train.info())
    # print(data_train.describe())

    # 船舱缺失处理
    data_train.loc[(data_train.Cabin.notnull()), 'Cabin'] = "Yes"
    data_train.loc[(data_train.Cabin.isnull()), 'Cabin'] = "No"


    # one_hot = [[1],[0]]
    # dict_vec = OneHotEncoder()
    # dict_vec.fit(one_hot)
    # X = dict_vec.transform(data_train)
    # print(X)




if __name__ == "__main__":
    # map_plt()
    # map2_plt()
    set_miss()
    pass

