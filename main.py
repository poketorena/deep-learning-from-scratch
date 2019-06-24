import numpy as np

x = np.array([0, 1])  # 入力
w = np.array([0.5, 0.5])  # 重み
b = -0.7  # バイアス
print(w * x)
print(np.sum(w * x))
print(np.sum(w * x) + b)
