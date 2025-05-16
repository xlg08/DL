"""
案例:
    演示损失函数之 二分类的损失函数.

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
    # 1. 定义真实值
    y_true = torch.tensor([0, 1, 0], dtype=torch.float)


    # 2. 定义预测值(概率)
    y_pred = torch.tensor([0.2901, 0.1432, 0.2412], requires_grad=True, dtype=torch.float)

    # 3. 定义二分类交叉熵损失.
    criterion = nn.BCELoss()

    # 4. 计算(二分类交叉熵)损失.
    loss = criterion(y_pred, y_true)

    # 5. 打印损失.
    print(f'二分类交叉熵损失: {loss}')


# 2. 测试
if __name__ == '__main__':
    dm01()