"""
案例:
    演示创建指定值的张量.

涉及到的函数:
    torch.ones 和 torch.ones_like 创建全1张量
    torch.zeros 和 torch.zeros_like 创建全0张量
    torch.full 和 torch.full_like 创建全为指定值张量

掌握:
    zeros(), ones(), full()
"""

# 导包
import torch
import numpy as np


# 需求1: torch.ones 和 torch.ones_like 创建全1张量
# 1. 创建2行4列全1张量
t1 = torch.ones(2, 4)
print(f't1: {t1}')  # [[1, 1, 1, 1], [1, 1, 1, 1]]

# 2. 按照数据形状, 创建全1张量.
t2 = torch.tensor([[1, 2], [4, 5]])     # 2行2列
print(f't2: {t2}')  # [[1, 2], [4, 5]]

t3 = torch.ones_like(t2)
print(f't3: {t3}')  # [[1, 1], [1, 1]]
print('-' * 24)


# 需求2: torch.zeros 和 torch.zeros_like 创建全0张量
# 1. 创建2行4列全1张量
t1 = torch.zeros(2, 4)
print(f't1: {t1}')  # [[1, 1, 1, 1], [1, 1, 1, 1]]

# 2. 按照数据形状, 创建全1张量.
t2 = torch.tensor([[1, 2], [4, 5]])     # 2行2列
print(f't2: {t2}')  # [[1, 2], [4, 5]]

t3 = torch.zeros_like(t2)
print(f't3: {t3}')  # [[1, 1], [1, 1]]
print('-' * 24)

# 需求3: torch.full 和 torch.full_like 创建全为指定值张量
# 1. 创建2行4列全1张量
t1 = torch.full(size=(2, 4), fill_value=66)
print(f't1: {t1}')  # [[66, 66, 66, 66], [66, 66, 66, 66]]

# 2. 按照数据形状, 创建全1张量.
t2 = torch.tensor([[1, 2], [4, 5]])     # 2行2列
print(f't2: {t2}')  # [[1, 2], [4, 5]]

t3 = torch.full_like(t2, fill_value=66)
print(f't3: {t3}')  # [[66, 66], [66, 66]]
print('-' * 24)