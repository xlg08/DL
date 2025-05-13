'''

'''

from torch import nn

# 参1：输入层个数
# 参2：输出层个数
# 参3：是否带有偏置
linear = nn.Linear(5, 3)
print(linear)       # Linear(in_features=5, out_features=3, bias=True)
print(linear.weight)
print(linear.weight.shape)      # (5, 3)
print(linear.weight.data)
print(type(linear.weight))  # linear.weight 为 <class 'torch.nn.parameter.Parameter'>  本质上是一个张量


print("--------------------权重初始化-----------------------")
# 权重初始化
# 0-1 均匀分布产生参数
nn.init.normal_(linear.weight)
print(linear.weight)

nn.init.zeros_(linear.weight)           # 将权重那种初始化为0
print(linear.weight)

# 正态分布随机初始化
nn.init.normal_(linear.weight, mean=0, std=1)       # 均值为0  标准差为1

print("*******************************************")
print(nn.init.ones_(linear.weight))