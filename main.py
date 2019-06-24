def AND(x1, x2):
    """
    単純パーセプトロンによるANDゲートの実装
    Args:
        x1:
        x2:

    Returns:
        ANDゲートの結果

    """

    # 重み1
    weight1 = 0.5

    # 重み2
    weight2 = 0.5

    # 閾値（閾値を超えたら発火）
    theta = 0.7

    tmp = x1 * weight1 + x2 * weight2

    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# mainプログラム
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
