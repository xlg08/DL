"""
案例:
    演示批量归一正则化.

批量归一正则化解释:
    概述:
        当数据量比较大的时候, 我们会分批次对数据进行训练, 但是如果批次之间的差异较大, 导致 模型参数 需要大范围调整 以便适应不同批次的"规律".
        这样就会导致 模型的收敛速度变慢, 甚至可能出现"梯度震荡, 梯度爆炸"
    解决思路:
        1. 对批次数据进行 标准化操作, 相当于减少了 批次数据之间的 差异性, 让数据更稳定.
        2. 弊端: 可能会丢失某些特征信息, 导致预测结果偏差, 所以引入了 λ(缩放, 当做: 权重) 和 β(平移, 当做: 偏置)
            f(x) = λ * (标准化后的x) + β

    批量归一化层, 常用的 3种方式:
        BatchNorm1d：主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为 (N, num_features) 的张量作为输入。
        BatchNorm2d：主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为 (N, C, H, W) 的张量作为输入。
        BatchNorm3d：主要用于三维卷积神经网络 (3D CNN)，处理三维数据，例如视频或医学图像。它接收形状为 (N, C, D, H, W) 的张量作为输入。
"""

# 导包
import torch
import torch.nn as nn

# 1. 定义函数, 演示: BatchNorm2d 处理图像. 例如: 输入 (N, C, H, W)
def dm01():
    # 1. 定义数据集 -> 充当输入数据, 模拟图片数据集.
    # 1张图片, 2个通道, 高:3像素, 宽:4像素.
    x = torch.rand(size=(1, 2, 3, 4))
    print(f'x: {x}')

    # 2. 搭建 批量归一化层.
    # 参1: 输入特征数, 等价于: 图片的 通道数.
    # 参2: 小常数, 防止处理时, 分母变为0.
    # 参3: 动量, 默认为: 0.1
    # 参4: 是否需要缩放, 默认为: True   是否计算 λ 和 β
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)

    # 3. 处理图像数据.
    y = bn2d(x)
    print(f'y: {y}')


# 2. 定义函数, 演示 BatchNorm1d 处理文本. 例如: 输入 (N, num_features)
def dm02():
    # 创建测试样本
    # 2个样本, 2个特征
    # 不能创建1个样本, 无法统计均值和方差4
    input_1d = torch.randn(size=(2, 2))
    print(f'input_1d: {input_1d}')

    # 创建线性层对象
    linear1 = nn.Linear(in_features=2, out_features=3)

    # 创建BN层对象
    # num_features：输入特征数
    bn1d = nn.BatchNorm1d(num_features=3)  # 20 output features
    output_1d = linear1(input_1d)

    # 进行批量归一化
    output = bn1d(output_1d)
    print("output-->", output)


# 3. 测试
if __name__ == '__main__':
    # dm01()
    dm02()