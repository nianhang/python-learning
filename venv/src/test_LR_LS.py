# -*- coding:utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print(np.random.seed(1234))
x = np.random.rand(500, 3)
y = x.dot(np.array([4.2, 5.7, 10.8]))

# 调用模型
lr = LinearRegression(fit_intercept=True)
# 训练模型
lr.fit(x,y)
print("估计的参数值为：%s" % (lr.coef_))
# 计算R平方
print('R2:%s' % (lr.score(x, y)))

# 任意设定变量，预测目标值
x_test = np.array([2, 4, 5]).reshape(1, -1)
print(x_test)
y_hat = lr.predict(x_test)
print("预测值为: %s" % (y_hat))



class LR_LS():
    def __init__(self):
        self.w = None
    def fit(self, X, y):
        # 最小二乘法矩阵求解
        #============================= show me your code =======================
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        print(self.w)
        #============================= show me your code =======================
    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        #============================= show me your code =======================
        y_pred = X.dot(self.w)
        #============================= show me your code =======================
        return y_pred

if __name__ == "__main__":
    lr_ls = LR_LS()
    lr_ls.fit(x,y)
    print("估计的参数值：%s" %(lr_ls.w))
    x_test = np.array([2, 4, 5]).reshape(1, -1)
    print("预测值为: %s" %(lr_ls.predict(x_test)))