import numpy as np

x = np.array([1, 2])
print(x)
print(x.shape)

w = np.array([[1, 3, 5], [2, 4, 6]])
print(w)
print(w.shape)

y = np.dot(x, w)
print(y)
print(y.shape)
