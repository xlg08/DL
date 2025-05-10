"""
案例:
    演示PyTorch中如何创建 线性 和 随机张量.

涉及到的函数:
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.random.initial_seed() 和 torch.random.manual_seed() 随机种子设置
    torch.rand/randn() 创建随机浮点类型张量
    torch.randint(low, high, size=()) 创建随机整数类型张量

掌握:
    arange(), manual_seed(), randn(), randint(),

"""
# 导包
import numpy as np
import torch


# 需求1: torch.arange() 和 torch.linspace() 创建线性张量
# 参1: 起始值, 参2: 结束值, 参3: 步长,  类似于: python的range(), numpy的 arange()
t1 = torch.arange(0, 10, 2)     # 包含起始值, 不包含结束值
print(f't1: {t1}')          # t1: tensor([0, 2, 4, 6, 8])

# 生成线性张量 -> 等差数列, 参1: 起始值, 参2: 结束值, 参3: 步长
t2 = torch.linspace(0, 10, 5)   # 包含起始值, 包含结束值
print(f't2: {t2}')
print('-' * 24)

# 需求2: torch.random.initial_seed() 和 torch.random.manual_seed() 随机种子设置
# print(torch.random.initial_seed())      # 基于时间戳来生成随机种子, 每次执行都不同.
# print(torch.random.manual_seed(24))       # 地址值: <torch._C.Generator object at 0x00000209FA99E970>

# 如果想要固定的 随机序列, 可以设置随机种子.
torch.random.manual_seed(24)

# 需求3: torch.rand/randn() 创建随机浮点类型张量
print(torch.rand(2, 3))     # 随机数, 0-1之间, 2行3列
print(torch.randn(2, 3))    # 正态分布
print('-' * 24)

# 需求4: torch.randint(low, high, size=()) 创建随机整数类型张量
print(torch.randint(2, 10, (2, 4)))