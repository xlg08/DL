"""
案例:
    演示梯度下降优化方法.

回顾: 之前旧版的 梯度更新公式
    W新 = W旧 - lr * 本次的梯度        可能出现: 平缓区更新慢, 鞍点, 局部最小值的问题

优化思路: 采用 指数移动加权平均
    指数移动加权平均:
        思路:
            计算所有历史梯度的平均值, 要结合权重, 越往前, 权重越小.
        公式:
            St = β*St-1 + (1 - β)*Yt
        解释:
            St:     本次的 移动加权平均值.
            St-1:   上次的 移动加权平均值.
            β:      权重调节系数, 越大, 看到的图(梯度)的效果越平缓.
            Yt:     本次的梯度
    更新公式为:
        W新 = W旧 - lr * St

结合 指数移动加权平均 和 梯度更新公式, 有了四种解决方案:
    动量法:            只从 梯度的方向优化, 给一个动量因子, 就是: β
        St = β*St-1 + (1 - β)*Yt
        W新 = W旧 - lr * St

    Adagrad:          是从 学习率 的角度优化.
        小常数: 1e-10
        St = St-1 + gt²
        lr = lr / (sqrt(St 累计平方梯度) + 小常数)

        W新 = W旧 - 处理后的lr * gt


    RMSProp:          是从 学习率 的角度优化.
        可以看做是 在Adagrad的基础上, 增加了 权重调和系数, 看计算学习率的时候, 更倾向于 历史的累计平方梯度 还是 本地的梯度平方和
        小常数: 1e-10
        St = β * St-1 + (1 - β) * gt²
        lr = lr / (sqrt(St 累计平方梯度) + 小常数)

        W新 = W旧 - 处理后的lr * gt

    Adam: 自适应矩估计            从 学习率 和 梯度两个角度优化.
        修正梯度 和 学习率, 即: Adam = 动量法 + RMSProp

        W新 = W旧 - 处理后的lr * 处理后的St
        需要传入两个β,  β1 -> 更新梯度,   β2 -> 更新累计平方梯度 -> 更新学习率.

总结:
    简单模型且数据量小:          SGD, Momentum(动量法)
    模型复杂且数据量大:          Adam
    需要处理稀疏数据或文本数据:   Adagrad, RMSProp
"""

# 导包
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义函数, 演示: 梯度优化方法_动量法
def dm01():
    # 1. 定义初始权重.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2. 定义损失函数.
    criterion = (w ** 2) / 2.0
    # 3. 定义优化器对象.
    # 参1: 要更新的参数列表.  参2: 学习率.  参3: 动量因子, 即: β值
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)
    # 4. 反向传播, 更新参数.
    # 梯度清零
    optimizer.zero_grad()
    # 计算梯度 -> 反向传播
    criterion.sum().backward()
    # 更新参数
    optimizer.step()
    # 5.打印更新后的参数值.
    print(f'w: {w}, w.grad: {w.grad}')

    # 第2次迭代:
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')


# 2. 定义函数, 演示: 梯度优化方法_Adagrad
def dm02():
    # 1. 定义初始权重.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2. 定义损失函数.
    criterion = (w ** 2) / 2.0
    # 3. 定义优化器对象.
    # 思路1: 动量法
    # 参1: 要更新的参数列表.  参2: 学习率.  参3: 动量因子, 即: β值
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    # 思路2: Adagrad(自适应的梯度)
    # 参1: 要更新的参数列表.  参2: 学习率.
    optimizer = optim.Adagrad(params=[w], lr=0.01)


    # 4. 反向传播, 更新参数.
    # 梯度清零
    optimizer.zero_grad()
    # 计算梯度 -> 反向传播
    criterion.sum().backward()
    # 更新参数
    optimizer.step()
    # 5.打印更新后的参数值.
    print(f'w: {w}, w.grad: {w.grad}')

    # 第2次迭代:
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')


# 3. 定义函数, 演示: 梯度优化方法_RMSProp
def dm03():
    # 1. 定义初始权重.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2. 定义损失函数.
    criterion = (w ** 2) / 2.0
    # 3. 定义优化器对象.
    # 思路1: 动量法
    # 参1: 要更新的参数列表.  参2: 学习率.  参3: 动量因子, 即: β值
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    # 思路2: Adagrad(自适应的梯度)
    # 参1: 要更新的参数列表.  参2: 学习率.
    # optimizer = optim.Adagrad(params=[w], lr=0.01)


    # 思路3: RMSProp(自适应的梯度)
    # 参1: 要更新的参数列表.  参2: 学习率.  参3: 指数衰减率(权重调和系数), 即: β值
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.9)

    # 4. 反向传播, 更新参数.
    # 梯度清零
    optimizer.zero_grad()
    # 计算梯度 -> 反向传播
    criterion.sum().backward()
    # 更新参数
    optimizer.step()
    # 5.打印更新后的参数值.
    print(f'w: {w}, w.grad: {w.grad}')

    # 第2次迭代:
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')


# 4. 定义函数, 演示: 梯度优化方法_Adam
def dm04():
    # 1. 定义初始权重.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2. 定义损失函数.
    criterion = (w ** 2) / 2.0
    # 3. 定义优化器对象.
    # 思路1: 动量法
    # 参1: 要更新的参数列表.  参2: 学习率.  参3: 动量因子, 即: β值
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    # 思路2: Adagrad(自适应的梯度)
    # 参1: 要更新的参数列表.  参2: 学习率.
    # optimizer = optim.Adagrad(params=[w], lr=0.01)


    # 思路3: RMSProp(自适应的梯度)
    # 参1: 要更新的参数列表.  参2: 学习率.  参3: 指数衰减率(权重调和系数), 即: β值
    # optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.9)


    # 思路4: Adam(自适应矩估计)
    # 参1: 要更新的参数列表.  参2: 学习率.  参3: 指数衰减率列表, β1 -> 更新梯度,  β2 -> 更新累计平方梯度 -> 更新学习率.
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))     # 大白话: betas[0] -> 动量法用的, betas[1] -> RMSProp用的

    # 4. 反向传播, 更新参数.
    # 梯度清零
    optimizer.zero_grad()
    # 计算梯度 -> 反向传播
    criterion.sum().backward()
    # 更新参数
    optimizer.step()
    # 5.打印更新后的参数值.
    print(f'w: {w}, w.grad: {w.grad}')

    # 第2次迭代:
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')



# 5. 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()