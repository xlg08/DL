"""
案例:
    创建指定类型的张量, 可以满足 用户灵活多变的需求.

涉及到的函数:
    data.type(torch.DoubleTensor)
    data.half/double/float/short/int/long()
"""

# 导包
import torch
import numpy as np


# 1. 定义张量, 指定类型为: int类型.
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
print(f't1: {t1}')
print(f't1.dtype: {t1.dtype}')  # t1张量中的 元素类型
print('-' * 24)

# 2. 思路1: type()函数, 转型.
t2 = t1.type(torch.float64)
print(t2, t2.dtype)

print(t1.type(torch.int32))
print(t1.type(torch.float32))       # torch.float32
print('-' * 24)


# 3. 思路2: data.half/double/float/short/int/long()
print(t1.half())        # torch.float16
print(t1.double())      # torch.float64
print(t1.float())       # torch.float32(默认)
print(t1.short())       # torch.int16
print(t1.int())         # torch.int32
print(t1.long())        # torch.int64(默认)
