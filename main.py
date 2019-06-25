import numpy as np
import matplotlib.pyplot as plt


# 微小な値に10^-4を用い、前方差分を用いる
def numerical_diff(f, x):
    h = 1e-4
    return ((f(x + h) - f(x - h)) / (2 * h))


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


x = np.arange(0.0, 20.0, 0.1)  # 0から20まで、0.1刻みのx配列
y = function_1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
tf1 = tangent_line(function_1, 5)
y2 = tf1(x)
plt.plot(x, y2)
tf2 = tangent_line(function_1, 12.5)
y3 = tf2(x)
plt.plot(x, y3)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 12.5))
