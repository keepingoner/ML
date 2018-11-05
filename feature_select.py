# -*- encoding:utf-8 -*-

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

"""特征处理 通过特定的统计方法或者数学方法，将数据转化为算法要求的数据
标准缩放；
1 /归一化
2/ 标准化

"""
# 使用较少，因为出现异常数据影响最大最小值
def min_max():
    """
    常用的方法是通过对原始数据进行线性变换把数据映射到[0,1]之间，
    变换的函数为：x = (x -min) / max -min
    :return:
    """
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[96, 68, 50, 20], [98, 79, 31, 54]])
    print(data)

# 标准化使用较多
def stand_vec():
    """
    常用的方法是z-score标准化，经过处理后的数据均值为0，标准差为1，处理方法是：
    x = (x - μ) / σ
​    μ是样本的均值，​σ 是样本的标准差
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[96, 68, 50, 20], [98, 79, 31, 54]])
    print(data)

# 数据抽取和分类
def dict_vec():
    """
    字典数据抽取
    :return:
    """
    a = [{"city": "beijing", "tmp": 100}, {"city": "shanghai", "tmp": 60}, {"city": "neimeng", "tmp": 10}]
    ab_dic = DictVectorizer(sparse=False)
    data = ab_dic.fit_transform(a)
    # 特征值的抽取
    print(ab_dic.get_feature_names())
    print(data)


def con_vec():
    """
    文本抽取
    :return:
    """
    a = ["i like python", 'python is short']

    ab_dic = CountVectorizer()

    data = ab_dic.fit_transform(a)

    print(ab_dic.get_feature_names())

    print(data.toarray())


def han_con_vec():
    """
    使用结巴进行汉字分词
    :return:
    """

    con1 = jieba.cut("我喜欢中国，我爱中国")
    con2 = jieba.cut("我喜欢python，我爱python")

    content1 = list(con1)
    content2 = list(con2)

    c1 = " ".join(content1)
    c2 = " ".join(content2)

    ab_dic = CountVectorizer()

    data = ab_dic.fit_transform([c1, c2])

    print(ab_dic.get_feature_names())

    print(data.toarray())


def tfidf_con_vec():
    """
    tf * idf 重要性
    :return:
    """

    c1, c2 = cut_word()

    ab_dic = TfidfVectorizer()

    data = ab_dic.fit_transform([c1, c2])

    # 特征值的抽取
    # print(ab_dic.get_feature_names())

    # 使用toarray方法
    print(data.toarray())


if __name__ == "__main__":

    # dict_vec()
    # con_vec()

    han_con_vec

    # tfidf_con_vec()




"""


二、标准化(Standardization)，或者去除均值和方差进行缩放
公式为：(X-X_mean)/X_std 计算时对每个属性/每列分别进行.
将数据按其属性(按列进行)减去其均值，然后除以其方差。
最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1。

sklearn.preprocessing.scale(X, axis=0, with_mean=True,with_std=True,copy=True)

根据参数的不同，可以沿任意轴标准化数据集。

参数解释：

X：数组或者矩阵
axis：int类型，初始值为0，axis用来计算均值 means 和标准方差 standard deviations. 如果是0，则单独的标准化每个特征（列），如果是1，则标准化每个观测样本（行）。
with_mean: boolean类型，默认为True，表示将数据均值规范到0
with_std: boolean类型，默认为True，表示将数据方差规范到1

X.mean(axis=0)用来计算数据X每个特征的均值；

X.std(axis=0)用来计算数据X每个特征的方差；

preprocessing.scale(X)直接标准化数据X。

"""

from sklearn import preprocessing
import numpy as np

X_train = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

X_scaled = preprocessing.scale(X_train)

# print(X_scaled)

X_mean = X_train.mean(axis=0)

X_std = X_train.std(axis=0)

X1 = (X_train - X_mean) / X_std

# print(X1)


# 方法2：sklearn.preprocessing.StandardScaler类


scaler = preprocessing.StandardScaler()

X_scaled2 = scaler.fit_transform(X_train)

# print(X_scaled2)

# X1 和X_trrain X_trrain2的值是相同的

"""

三、将特征的取值缩小到一个范围（如0到1）
除了上述介绍的方法之外，另一种常用的方法是将属性缩放到一个指定的最大值和最小值(通常是1-0)之间，这可以通过preprocessing.MinMaxScaler类来实现。

使用这种方法的目的包括：

1、对于方差非常小的属性可以增强其稳定性；
2、维持稀疏矩阵中为0的条目。
下面将数据缩至0-1之间，采用MinMaxScaler函数

"""
min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(X_train)

# print(X_train_minmax)
#
# print(min_max_scaler.scale_)
#
# print(min_max_scaler.min_)

"""
注意：这些变换都是对列进行处理。

当然，在构造类对象的时候也可以直接指定最大最小值的范围：feature_range=(min, max)，此时应用的公式变为：

"""

X_std = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))

X_minmax = X_std / (X_train.max(axis=0) - X_train.min(axis=0)) + X_train.min(axis=0)

"""
四、正则化(Normalization)
正则化的过程是将每个样本缩放到单位范数(每个样本的范数为1)，如果要使用如二次型(点积)或者其它核方法计算两个样本之间的相似性这个方法会很有用。

该方法是文本分类和聚类分析中经常使用的向量空间模型（Vector Space Model)的基础.

Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数(l1-norm,l2-norm)等于1。

"""
# 方法1：使用sklearn.preprocessing.normalize()函数

X_normalized = preprocessing.normalize(X_train, norm='l2')

# print(X_normalized)

# 方法2：sklearn.preprocessing.StandardScaler类
normalizer = preprocessing.Normalizer().fit(X_train)
# 然后使用正则化实例来转换样本向量：

# print(normalizer.transform(X_train))

# 两种方法都可以，效果是一样的。


"""
五、二值化(Binarization)

特征的二值化主要是为了将数据特征转变成boolean变量。在sklearn中，sklearn.preprocessing.Binarizer函数可以实现这一功能。实例如下：

"""

binarizer = preprocessing.Binarizer().fit(X_train)

# 带参数 Binarizer(threshold=1.1)   threshold为阀值 结果数据值大于阈值的为1，小于阈值的为0


binarizer.transform(X_train)

# print(binarizer.transform(X_train))


"""
六、缺失值处理
由于不同的原因，许多现实中的数据集都包含有缺失值，要么是空白的，要么使用NaNs或者其它的符号替代。
这些数据无法直接使用scikit-learn分类器直接训练，所以需要进行处理。
幸运地是，sklearn中的Imputer类提供了一些基本的方法来处理缺失值，如使用均值、中位值或者缺失值所在列中频繁出现的值来替换。


"""

from sklearn.preprocessing import Imputer

X = [[1, 2], [np.nan, 3], [7, 6]]

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

data = imp.fit_transform(X)

print(data)

X = [[np.nan, 2], [6, np.nan], [7, 6]]

print(imp.transform(X))



from sklearn.feature_extraction import DictVectorizer



"""

将映射列表转换为Numpy数组或scipy.sparse矩阵  

sklearn.feature_extraction.DictVectorizer(sparse = True)
sparse 是否转换为scipy.sparse矩阵表示，默认开启

方法
fit_transform(X,y)

应用并转化映射列表X，y为目标类型

inverse_transform(X[, dict_type])

将Numpy数组或scipy.sparse矩阵转换为映射列表

toarray() 和  sparse=False  使用一个就好
"""

onehot = DictVectorizer()

instances = [{'city': '北京', 'temperature': 100}, { 'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]

X = onehot.fit_transform(instances).toarray()

# print(onehot.inverse_transform(X))

# print(X)

# 查看特征名字
# print(onehot.get_feature_names())



"""
文本特征抽取


"""

from sklearn.feature_extraction.text import CountVectorizer

content = ["life is short,i like python","life is too long,i dislike python"]

vectorizer = CountVectorizer()

# print(vectorizer.fit_transform(content).toarray())

# print(vectorizer.get_feature_names())


"""
tf*idf 词的重要性
"""

from sklearn.feature_extraction.text import TfidfVectorizer

content = ["life is short,i like python", "life is too long,i dislike python"]

vectorizer = TfidfVectorizer(stop_words='english')

print(vectorizer.fit_transform(content).toarray())

print(vectorizer.vocabulary_)

print(vectorizer.get_feature_names())





"""
降维  降的是特征数量

1/特征选择
2/主成分分析 PCA

n_components 小数   90% -95% 最好 经验
n_components 整数 减少到的特征数量  一般不使用整数

特征选择三大武器  1 过滤 variance 方差为0  基本不变化 2 嵌入式 （正则、决策树） 3 包裹

"""

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# 使用较少，因为出现异常数据影响最大最小值


def min_max():
    """
    删除低方差的特征
    :return:
    """

    mm = VarianceThreshold(threshold=0)

    data = mm.fit_transform([[96, 68, 50, 20], [98, 79, 31, 54]])

    print(data)


def main_sele():
    """
    主成分分析
    :return:
    """
    mm = PCA(n_components=0.9)

    data = mm.fit_transform([[96, 68, 50, 20], [98, 79, 31, 54]])

    print(data)


if __name__ == "__main__":

    # min_max()
    main_sele()

