"""
案例:
    演示 dropout正则化.

dropout正则化解释:
    概述:
        让神经元以p的概率 随机失活, 防止模型训练时过度依赖某些特征.
        p的的值一般是 [0.2, 0.5], 如果是简单模型, 可以小一点, 如果是复杂模型, 可以大一点.
    细节:
        1. 未被失活的数据(神经元)会被进行缩放, 缩放比例是:  1/(1-p)
        2. dropout一般应用于 激活层之后, 即: 加权求和 -> 激活函数 -> dropout
        3. dropout只应用于 训练阶段, 在测试阶段失效.
            设置模型状态的两个函数:
                model.train() -> 训练阶段
                model.eval() -> 测试阶段
"""

# 导包
import torch
import torch.nn as nn

# 1. 准备数据集.
# 随机生成 [1, 10]之间的数据, 形状为: 1行4列, 即: 1条样本4个特征
x = torch.randint(1, 10, (1, 4), dtype=torch.float)

# 2. 创建隐藏层
linear1 = nn.Linear(4, 5)
# print(linear1.weight)       # 权重矩阵

# 3. 先对特征进行 加权求和处理, 再进行 激活函数(激活层)处理.
x = linear1(x)
x = torch.relu(x)  # relu激活函数, 只考虑正样本, max(0, x)
print(f'未被失活的数据: {x}')

# 4. 创建随机失活层, 设置p值(每个神经元随机失活的概率)
dp = nn.Dropout(p=0.5)
# 对 激活函数处理后的结果 进行 随机失活.
out = dp(x)
print(f'被失活的数据: {out}')