"""
案例:
    演示下池化层的API.

池化层介绍:
    用于对 卷积层处理后的特征图 进行降维的, 有 最大池化 和 平均池化两种方式, 推荐使用 最大池化.

    池化层不会改变通道数, 仅仅是 降维.
"""

import torch
import torch.nn as nn


# 场景1: 演示 单通道池化
def dm01():
    # 1. 定义输入数据 [1,3,3]
    x = torch.tensor([
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ],
    ])
    print(x.shape)      # torch.Size([1, 3, 3])

    # 2. 定义最大池化层.
    # 参1: 池化核的大小, 参2: 步长, 参3: 填充
    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    # 3. 最大池化操作.
    out1 = max_pool(x)
    print(f'out1: {out1}')
    print(f'out1.shape: {out1.shape}')  # torch.Size([1, 2, 2])
    print('-' * 24)

    # 4. 定义平均池化层.
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    out2 = avg_pool(x)
    print(f'out2: {out2}')
    print(f'out2.shape: {out2.shape}')  # torch.Size([1, 2, 2])


# 场景2: 演示 多通道池化
def dm02():
    # 定义输入数据 [3,3,3]
    inputs = torch.tensor([
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ],

        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ],

        [
            [11, 22, 33],
            [44, 55, 66],
            [77, 88, 99]
        ]
    ], dtype=torch.float)
    # 最大池化
    pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = pooling(inputs)
    print("多通道池化：\n", output)

# 测试
if __name__ == '__main__':
    # dm01()
    dm02()