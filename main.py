import numpy as np


# 微小な値に10^-4を用い、前方差分を用いる
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h) / (2 * h))


# 丸目誤差で正しく計算できない！
print(np.float32(1e-50))
