"""
案例:
    演示卷积层 相关API.
"""

# 导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 加载图像.
img1 = plt.imread('../../data/img.jpg')     # HWC -> 640 * 640 * 3
print(f'img1: {img1.shape}')            # (640, 640, 3)

# 2. 把图转成 张量.
img2 = torch.tensor(img1, dtype=torch.float32)
print(f'img2: {img2.shape}')            # torch.Size([640, 640, 3])

# 3. 把图像从 HWC -> CHW
img3 = img2.permute(dims=(2, 0, 1))
print(f'img3: {img3.shape}')            # torch.Size([3, 640, 640])

# 4. 把图像转成4维张量, 即: 1张图 3通道 640 * 640
img4 = img3.unsqueeze(dim=0)
print(f'img4: {img4.shape}')            # torch.Size([1, 3, 640, 640])

# 语句合并
# img4 = torch.tensor(plt.imread('../../data/img.jpg'), dtype=torch.float32).permute(dims=(2, 0, 1)).unsqueeze(dim=0)


# 5. 创建 卷积层.
# 参1: 输入通道数 = 图片的通道数, 参2: 输出通道数, 参3: 卷积核大小, 参4: 步长, 参5: 填充.
conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)

# 6. 卷积处理, 从图像中自动提取信息.
conv_img = conv(img4)
print(f'conv_img: {conv_img.shape}')

# 7. 从卷积层处理后的内容中, 获取到具体的图片.
img5 = conv_img[0].permute(dims=(1, 2, 0))          # [4, 638, 638] -> [638, 638, 4]
print(f'img5: {img5.shape}')            # torch.Size([638, 638, 4])

# 8. 绘制处理后的 4张 特征图中的 第1张图.    记得转成numpy, 然后显式.
plt.imshow(img5[:, :, 0].detach().numpy())
plt.show()