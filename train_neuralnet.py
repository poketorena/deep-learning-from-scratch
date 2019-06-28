import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask_train = np.random.choice(train_size, batch_size)
    x_train_batch = x_train[batch_mask_train]
    t_train_batch = t_train[batch_mask_train]
    batch_mask_test = np.random.choice(test_size, batch_size)
    x_test_batch = x_test[batch_mask_test]
    t_test_batch = t_test[batch_mask_test]

    # 誤差逆伝播法によって勾配を求める
    grad = network.gradient(x_train_batch, t_train_batch)

    # 更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    train_loss = network.loss(x_train_batch, t_train_batch)
    train_loss_list.append(train_loss)
    test_loss = network.loss(x_test_batch, t_test_batch)
    test_loss_list.append(test_loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(
            f"train_loss:{train_loss:.4f} test_loss:{test_loss:.4f}  train_acc:{train_acc:.4f} test_acc:{test_acc:.4f}")

# 結果をグラフで可視化する
plt.title("loss graph")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.plot(range(len(train_loss_list)), train_loss_list, label="train loss")
plt.plot(range(len(test_loss_list)), test_loss_list, label="test loss")
plt.legend()
plt.show()

plt.title("accuracy graph")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(range(len(train_acc_list)), train_acc_list, label="train acc")
plt.plot(range(len(test_acc_list)), test_acc_list, label="test acc")
plt.legend()
plt.show()
