"""
案例:
    演示 指数移动加权平均.

使用梯度下降优化方法时, 可能遇到的问题:
    公式:
        W新 = W旧 - lr * grad
    可能遇到的问题:
        1. 数据比较平缓, 梯度值较小, 迭代慢.
        2. 遇到鞍点, 导为0, 可能会认为"最优值"
        3. 可能存在 局部最小值.

解决方案:
    动量法, Momentum
    Adagrad
    RMSProp
    Adam

    但是上述的这四种方式, 都要依赖: 指数移动加权平均.

指数移动加权平均解释:
    大白话:
        预测明天的天气, 和今天天气的关系比较大, 和上月的天气关系较小, 越往前, 权重越低.
    专业:
        基于所有的历史梯度, 计算: 加权平均值. 原则: 越往前, 权重越小.
    公式:
        St = β*St-1 + (1 - β)*Yt
    解释:
        St:     本次的 移动加权平均值.
        St-1:   上次的 移动加权平均值.
        β:      权重调节系数, 越大, 看到的图(梯度)的效果越平缓.
        Yt:     本次的梯度
"""

import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30


# 1. 实际平均温度
def dm01():
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.show()


# 2. 指数加权平均温度
def dm02(beta=0.9):
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)

    exp_weight_avg = []
    # idx从1开始
    for idx, temp in enumerate(temperature, 1):
        # 第一个元素的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
        # idx-2：2-2=0，exp_weight_avg列表中第一个值的下标值
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color='r')
    plt.scatter(days, temperature)
    plt.show()


if __name__ == '__main__':
    dm01()
    dm02(0.5)
    dm02(0.9)