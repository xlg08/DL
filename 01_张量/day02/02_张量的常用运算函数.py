"""
    案例:
        演示常量的常用运算函数.

    涉及到的函数如下:
        sum()   求和              掌握
        max()   最大值
        min()   最小值
        mean()  平均值

        pow()   指定次幂
        sqrt()  平方根

        exp()   e的n次幂

        log()   以e为底
        log2()  以2为底
        log10() 以10为底
"""

# 导包
import torch

# 1. 定义张量, 浮点型.
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
print(f't1: {t1}')

# 2. 求和, 有dim属性, 0: 列, 1: 行
print(t1.sum(dim=0))    # 按列求和
print(t1.sum(dim=1))    # 按行求和
print(t1.sum())         # 全局求和

print(t1.mean())        # 全局求平均值, 必须是: 浮点型.
print('-' * 24)

# 3. 求平方,立方 和 平方根, 没有dim属性.
print(t1.pow(2))    # 平方
print(t1.pow(3))    # 立方
print(t1.sqrt())    # 平方根
print('-' * 24)

# 4. e的n次幂, 没有dim属性.
print(t1.exp()) # e^1, e^2, e^3, ..., e^6
print('-' * 24)

# 5. 求对数, 没有dim属性.
print(t1.log())     # 以e为底的对数, ln(1), ln(2), ln(3), ..., ln(6)
print(t1.log2())    # 以2为底的对数, log2(1), log2(2), log2(3), ..., log2(6)
print(t1.log10())   # 以10为底的对数, log10(1), log10(2), log10(3), ..., log10(6)
