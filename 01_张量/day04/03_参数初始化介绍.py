"""
案例:
    演示下参数初始化相关内容.

参数初始化相关概述:
    解释:
        就是通过 API 来初始化 权重矩阵(Weight) 和 偏置矩阵(Bias)
    目的/作用:
        1. 防止梯度消失 或者 梯度爆炸.
        2. 加快(提高)模型收敛速度.
        3. 打破数据对称性, 类似于线性回归增加 噪声.
    方式:
        1. 均匀分布随机初始化
        2. 固定初始化
        3. 全0初始化
        4. 全1初始化
        5. 正态分布随机初始化
        6. kaiming初始化
        7. xavier初始化
    如何选择?
        深层, 浅层均可:
            Relu            -> kaiming
            Tanh,Sigmoid    -> xavier
        如果是浅层, 还可以用 随机初始化 normal_()
"""

import torch.nn as nn


# 1. 均匀分布随机初始化
def dm01():
    # 1. 创建隐藏层1 -> 输入特征: 5, 输出特征: 3
    linear1 = nn.Linear(5, 3)
    # 2. 初始化隐藏1的参数信息 -> 权重矩阵, 偏置矩阵
    nn.init.uniform_(linear1.weight)
    # 3. 打印隐藏层1的参数信息
    print(f'linear1.weight: {linear1.weight}')  # 权重矩阵
    # print(f'linear1.bias: {linear1.bias}')      # 偏置矩阵

# 2. 固定初始化
def dm02():
    # 1. 创建隐藏层1 -> 输入特征: 5, 输出特征: 3
    linear1 = nn.Linear(5, 3)
    # 2. 初始化隐藏1的参数信息 -> 权重矩阵, 偏置矩阵
    nn.init.constant_(linear1.weight, 2.4)

    # 3. 打印隐藏层1的参数信息
    print(f'linear1.weight: {linear1.weight}')  # 权重矩阵


# 3. 全0初始化
def dm03():
    # 1. 创建隐藏层1 -> 输入特征: 5, 输出特征: 3
    linear1 = nn.Linear(5, 3)
    # 2. 初始化隐藏1的参数信息 -> 权重矩阵, 偏置矩阵
    nn.init.zeros_(linear1.weight)
    # 3. 打印隐藏层1的参数信息
    print(f'linear1.weight: {linear1.weight}')  # 权重矩阵


# 4. 全1初始化
def dm04():
    # 1. 创建隐藏层1 -> 输入特征: 5, 输出特征: 3
    linear1 = nn.Linear(5, 3)
    # 2. 初始化隐藏1的参数信息 -> 权重矩阵, 偏置矩阵
    nn.init.ones_(linear1.weight)
    # 3. 打印隐藏层1的参数信息
    print(f'linear1.weight: {linear1.weight}')  # 权重矩阵


# 5. 正态分布随机初始化
def dm05():
    # 1. 创建隐藏层1 -> 输入特征: 5, 输出特征: 3
    linear1 = nn.Linear(5, 3)
    # 2. 初始化隐藏1的参数信息 -> 权重矩阵, 偏置矩阵
    nn.init.normal_(linear1.weight, mean=0, std=1) # 均值0, 标准差1
    # 3. 打印隐藏层1的参数信息
    print(f'linear1.weight: {linear1.weight}')  # 权重矩阵

# 6. kaiming 初始化
def dm06():
    # 1. 创建隐藏层1 -> 输入特征: 5, 输出特征: 3
    linear1 = nn.Linear(5, 3)

    # 思路1: 正态分布初始化
    # 2. 初始化隐藏1的参数信息 -> 权重矩阵, 偏置矩阵
    # 参1: 权重矩阵, 参2: 激活函数名称 -> 默认是Leaky Relu, 可以修改为: Relu
    nn.init.kaiming_normal_(linear1.weight, nonlinearity='relu')
    # 3. 打印隐藏层1的参数信息
    print(f'linear1.weight: {linear1.weight}')  # 权重矩阵

    # 思路2: 均匀分布初始化
    nn.init.kaiming_uniform_(linear1.weight)
    print(f'linear1.weight: {linear1.weight}')


# 7. xavier 初始化
def dm07():
    # 1. 创建隐藏层1 -> 输入特征: 5, 输出特征: 3
    linear1 = nn.Linear(5, 3)

    # 思路1: xavier 正态分布初始化
    # 2. 初始化隐藏1的参数信息 -> 权重矩阵, 偏置矩阵
    nn.init.xavier_normal_(linear1.weight)
    # 3. 打印隐藏层1的参数信息
    print(f'linear1.weight: {linear1.weight}')  # 权重矩阵

    # 思路2: xavier 均匀分布初始化
    nn.init.xavier_uniform_(linear1.weight)
    print(f'linear1.weight: {linear1.weight}')



if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    # dm06()
    dm07()