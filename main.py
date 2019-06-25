import numpy as np


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


data = np.array([0.3, 2.9, 4.0])
result = softmax(data)
print(result)
