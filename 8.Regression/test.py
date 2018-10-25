# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('8.Advertising.csv')

x = data[['TV', 'Radio', 'Newspaper']]

y = data['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# model = Lasso()
model = Ridge()

alpha_can = np.logspace(-3, 2, 10)
print(alpha_can)

lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)

lasso_model.fit(x, y)

print '验证参数：\n', lasso_model.best_params_

y_hat = lasso_model.predict(np.array(x_test))

mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error

rmse = np.sqrt(mse)  # Root Mean Squared Error

print mse, rmse

# 预测准确率
print("预测准确率{}".format(lasso_model.score(x_test, y_test)))

# 交叉验证中最好的结果

print("交叉验证中最好的结果{}".format(lasso_model.best_score_))

# 最好的模型

print("最好的模型{}".format(lasso_model.best_estimator_))

# 每个k的验证结果

print("每个k的验证结果{}".format(lasso_model.cv_results_))


t = np.arange(len(x_test))

plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()






