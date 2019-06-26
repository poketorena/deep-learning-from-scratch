import numpy as np


def function_2(x):
    return x[0] ** 2 + x[1] ** 2
    # または return np.sum(x**2)


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 値を元に戻す

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(result)

# 学習率が大きすぎる例
init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr=10, step_num=100)
print(result)

# 学習率が小さすぎる例
init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
print(result)
