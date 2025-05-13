"""
案例:
    演示如何创建张量.

涉及到的函数如下:
    torch.tensor 根据指定数据创建张量
    torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

张量解释:
    在PyTorch框架中, 所有的数据(标量, 矢量, 向量)都会被封装成 张量(Tensor).
    PyTorch是深度学习框架, 属于第三方包, 用之前需要 安装一下.
        pip install torch

细节:
    Pytorch中的张量必须是 数值型.

掌握:
    torch.tensor 根据指定数据创建张量

"""

# 导包
import torch
import numpy as np

# 场景1: torch.tensor 根据指定数据创建张量
# 需求1: 封装 标量(0维, 1个值)
# t1 = torch.tensor('你好')         # 报错.
# t1 = torch.tensor(24)
t1 = torch.tensor(True)
print(f't1: {t1}')
print(f't1的值: {t1.item()}')
print(f't1的元素类型: {t1.dtype}')
print(f't1的形状: {t1.shape}')
print('-' * 24)

# 需求2: 封装 矢量(1维, 多个值)
t2 = torch.tensor([1, 2, 3, 4, 5])
print(f't2: {t2}')
print(f't2的元素类型: {t2.dtype}')
print(f't2的形状: {t2.shape}')
print('-' * 24)

# 需求3: 封装 向量(2维, 多个值), 例如: 2行3列.
data = [[1, 2, 3], [4, 5, 6]]
# t3 = torch.Tensor(data)                   # 默认是浮点型float32
t3 = torch.tensor(data, dtype=torch.int)    # 指定数据类型为: int32
print(f't3: {t3}')
print(f't3的元素类型: {t3.dtype}')
print(f't3的形状: {t3.shape}')
print('-' * 24)

# 需求4: 把 numpy的 ndarray封装成 张量(tensor)
# data = np.arange(0, 10).reshape(2, 5)
data = np.random.randint(1, 10, size=(2, 3))
# print(data, type(data))
t4 = torch.tensor(data)
print(f't4: {t4}')
print(f't4的元素类型: {t4.dtype}')
print(f't4的形状: {t4.shape}')
print('-' * 24)


# 需求5: 尝试用 指定维度的方式 来创建张量, 发现: torch.tensor()不支持.
# t5 = torch.tensor(2, 3)       # 报错.
# print(f't5: {t5}')
print('*' * 24)


# 场景2: torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
# 需求1: 封装 标量(0维, 1个值)
t1 = torch.Tensor(24)
print(f't1: {t1}')
print(f't1的元素类型: {t1.dtype}')
print(f't1的形状: {t1.shape}')
print('-' * 24)

# 需求2: 封装 矢量(1维, 多个值)
t2 = torch.Tensor([1, 2, 3, 4, 5])
print(f't2: {t2}')
print(f't2的元素类型: {t2.dtype}')
print(f't2的形状: {t2.shape}')
print('-' * 24)

# 需求3: 封装 向量(2维, 多个值), 例如: 2行3列.
data = [[1, 2, 3], [4, 5, 6]]
t3 = torch.Tensor(data)                   # 默认是浮点型float32
print(f't3: {t3}')
print(f't3的元素类型: {t3.dtype}')
print(f't3的形状: {t3.shape}')
print('-' * 24)

# 需求4: 把 numpy的 ndarray封装成 张量(tensor)
# data = np.arange(0, 10).reshape(2, 5)
data = np.random.randint(1, 10, size=(2, 3))
# print(data, type(data))
t4 = torch.Tensor(data)
print(f't4: {t4}')
print(f't4的元素类型: {t4.dtype}')
print(f't4的形状: {t4.shape}')
print('-' * 24)

# 需求5: 指定维度创建张量.
t5 = torch.Tensor(2, 3)
print(f't5: {t5}')
print('*' * 24)


# 场景3: torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
print(torch.IntTensor(10))  # 这里的10不是数据, 而是形状. 会生成一个10个元素的张量.
print(torch.IntTensor([10]))
print(torch.IntTensor([1, 2, 3]))
print(torch.IntTensor([[1, 2, 3], [4, 5, 6]]))
print(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]))    # 会做类型转换





