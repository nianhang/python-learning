# -*- coding:utf-8 -*-
import numpy as np
# 1
# 2 版本和配置
print(np.__version__)
np.show_config()

# 3 长度为10的空向量
Z = np.zeros(10)
print(Z)

#4
A = np.zeros((10,10))
#print(A)
#print(A.size)
#print(A.itemsize)
print("%d bytes" % (A.size * A.itemsize))

#5
np.info(np.add)

#6
Z6 = np.zeros(10)
Z6[4] = 1
print(Z6)

#7
Z7 = np.arange(10,50)
print(Z7)

#8
z8 = np.arange(10)
z8 = z8[::-1]
print(z8)

#9
z9 =np.arange(9).reshape(3,3)
print(z9)

#10
nz10 = np.nonzero([1,2,0,0,4,0])
print(nz10)

#11
z11 = np.eye(3)
print(z11)

#12
z12 = np.random.random((3,3,3))
print(z12)

#13
z13 = np.random.random((10,10))
z13min,z13max = z13.min(),z13.max()
print(z13,z13max,z13min)

#14
z14 = np.random.random(30)
z14mean = z14.mean()
print(z14mean)

#15
z15 = np.zeros((10,10))
print(z15)
z15[0,:] = 1
z15[-1,:] = 1
z15[:,0] = 1
z15[:,-1] = 1
print(z15)

z15h = np.ones((10,10))
z15h[1:-1,1:-1] = 0
print(z15h)

#16
z16 = np.ones((5,5))
print(z16)
z16 = np.pad(z16, pad_width=1, mode='constant', constant_values=0)
print(z16)

#17
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)

#18
z18 = np.diag(1 + np.arange(4), k=-1)
print(z18)
# 对角线的位置
z18h = np.diag(1+np.arange(5))
print(z18h)

#19
z19 = np.zeros((8, 8), dtype=int)
z19[1::2, ::2] = 1
z19[::2, 1::2] = 1
print(z19)

#20
z20 = np.arange(336).reshape(6, 7, 8)
#print(z20)
print(np.unravel_index(100, (6, 7, 8)))

# A = np.random.randint(1, 100, size=(3, 3, 3, 2))
# ind_max = np.argmax(A)
# ind_max_src = np.unravel_index(ind_max, A.shape)
# print(A)
# print(ind_max)
# print(ind_max_src)
# print(A[ind_max_src])

#21
z21 = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(z21)

#22
z22 = np.random.random((5, 5))
z22max, z22min = z22.max(), z22.min()
z22 = (z22 - z22min)/(z22max - z22min)
print(z22)

#23
# color = np.dtype([("r", np.ubyte, 1),
#                   ("g", np.ubyte, 1),
#                   ("b", np.ubyte, 1),
#                   ("a", np.ubyte, 1)])
# color

#24
z24 = np.dot(np.ones((5, 3)), np.ones((3, 2)))
print(z24)

#25
z25 = np.arange(11)
z25[(3 < z25) & (z25 <= 8)] *= -1
print(z25)

#26
print(sum(range(5), -1))
from numpy import *
print(sum(range(5), -1))

#27
z27 = np.arange(5)
print(z27 ** z27)
print(z27 < - z27)
print(1j * z27)
print(z27/1/1)
print(z27 * z27)

#28
#print(np.array(0) / np.array(0))
#print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))

#29
z29 = np.random.uniform(-10, +10, 10)
print(z29)
print(np.copysign(np.ceil(np.abs(z29)), z29))

#30 求两个数组的交集
z30_1 = np.random.randint(0, 10, 10)
z30_2 = np.random.randint(0, 10, 10)
print(z30_1)
print(z30_2)
print(np.intersect1d(z30_1, z30_2))
