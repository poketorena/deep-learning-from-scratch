import numpy as np


# 悪い実装例
def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x) / h)


# 丸目誤差で正しく計算できない！
print(np.float32(1e-50))
