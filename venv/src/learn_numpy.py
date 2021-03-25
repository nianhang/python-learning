import numpy as np
from numpy import pi

a = np.arange(15).reshape(3,5)

# print(a)
#
# print(type(a))
#
# print(a.size)
#
# print(a.dtype.name)
# print(a.itemsize)

b = np.array([1.2,3.5,5.1])
#print(b.dtype)

c = np.empty((2,3))
#print(c)

d = np.linspace(0,2*pi,100)
f = np.sin(d)
print(f)

m = np.arange(24).reshape(2,3,4)
print(m)

print(np.arange(10000))
