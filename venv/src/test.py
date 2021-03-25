# -*- coding:utf-8 -*-
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 定义目标函数通过改函数产生对应的y
# y=1*x[0]+2*x[1]+....(n+1)*x[n]
def l_model(x):
    params = np.arange(1, x.shape[-1]+1)
    y = np.sum(params*x) + np.random.randn(1)*0.1  #增加一点随机变量
    return y

# 定义数据集
x = pd.DataFrame(np.random.randn(500, 6))
# print(x.shape[-1])
# print(x.shape[0])
# print(x.shape[1])
y = x.apply(lambda x_rows: pd.Series(l_model(x_rows)), axis = 1)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2)

# 数据标准化
ss = StandardScaler()
x_train_s = ss.fit_transform(x_train)
x_test_s = ss.transform(x_test)

print(ss.scale_)
print(ss.mean_)

# 训练模型
lr = LinearRegression()
lr.fit(x_train_s, y_train)

print(lr.coef_)
print(lr.intercept_)

y_predict = lr.predict(x_test_s)
lr.score(x_test_s, y_test)

## 预测值和实际值画图比较
t = np.arange(len(x_test_s))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_predict, 'b+', linewidth=1, label='预测值')
plt.legend(loc='upper left') #显示图例，设置图例的位置
plt.title("线性回归预测真实值之间的关系", fontsize=20)
plt.grid(b=True)#加网格
plt.show()

## 预测值和实际值画图比较
# t=np.arange(len(x_test_s))
# plt.figure(facecolor='w')#建一个画布，facecolor是背景色
# plt.plot(t, y_test, 'r-', linewidth=2, label='真实值')
# plt.plot(t, y_predict, 'g-', linewidth=1, label='预测值')
# plt.legend(loc = 'upper left')#显示图例，设置图例的位置
# plt.title("线性回归预测真实值之间的关系", fontsize=20)
# plt.grid(b=True)#加网格
# plt.show()







