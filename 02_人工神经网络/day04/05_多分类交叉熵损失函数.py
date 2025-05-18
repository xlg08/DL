"""
案例:
    演示损失函数之 多分类的损失函数.

损失函数介绍:
    概述/作用:
        用来评估模型好坏的, 即: 模型的拟合情况.
    分类:
        回归损失函数:
            MAE
            MSE
            Smooth L1
        分类损失函数:
            二分类损失函数:    BCELoss
            多分类损失函数:    CrossEntropyLoss = Softmax + 损失计算

            记忆:
                无论是二分类损失函数, 还是多分类损失函数, API底层默认都是算 平均损失(mean)
"""

# 需求: 演示下 多分类损失函数的应用.

# 导包
import torch
import torch.nn as nn


# 1. 定义函数, 演示: 多分类交叉熵损失函数.
def dm01():

    # 1. 定义真实值, 可以是: one-hot, 也可以是: 索引.
    # 热(one-hot)编码处理后的值.
    # y_true = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float)   # 第1个样本: B类, 第2个样本: C类

    # 索引值.
    y_true = torch.tensor([1, 2], dtype=torch.int64)

    # 2. 定义预测值(概率)
    y_pred = torch.tensor([[0.2, 0.6, 0.2], [0.1, 0.1, 0.8]], requires_grad=True, dtype=torch.float)

    # 3. 定义多分类交叉熵损失.
    criterion = nn.CrossEntropyLoss()   # 底层 = Softmax + 损失计算.

    # 4. 计算(多分类交叉熵)损失.
    loss = criterion(y_pred, y_true)

    # 5. 打印损失.
    print(f'多分类交叉熵损失: {loss}')


# 2. 测试
if __name__ == '__main__':
    dm01()