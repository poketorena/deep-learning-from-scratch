import numpy as np
import matplotlib.pyplot as plt


# 微小な値に10^-4を用い、前方差分を用いる
def numerical_diff(f, x):
    h = 1e-4
    return ((f(x + h) - f(x - h)) / (2 * h))


# x0=3、x1=4の時のx0に対する偏微分を求めよ
def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2


# x0=3、x1=4の時のx1に対する偏微分を求めよ
def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 * x1


print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))
