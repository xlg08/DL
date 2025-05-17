"""
案例:
    演示图像的基本绘制.

图像分类:
    二值图:    1通道, 像素: 0, 1
    灰度图:    1通道, 像素: 0~255
    索引图:    1通道, 像素: 0~255 + 索引对应的 颜色矩阵.
    彩色图:    3通道, 像素: 0~255, 这个是以后我们用的最多的数据格式.

涉及到的API:
    imread()     读取图片, 获取: 像素矩阵.
    imsave()     保存图片, 保存: 像素矩阵.
    imshow()     绘制图片, 显示: 像素矩阵.
"""

# 导包
import matplotlib.pyplot as plt
import numpy as np


# 1. 绘制 全黑图 和 全白图.
def dm01():
    # 场景1: 绘制纯黑图, 像素值越接近0, 越黑, 越接近255, 越白.
    # 像素: HWC -> 高 宽 通道
    img1 = np.zeros(shape=(200, 200, 3))
    # print(f'img1: {img1}')
    # print(f'img1.shape: {img1.shape}')

    plt.imshow(img1)
    # plt.axis('off') # 关闭坐标系
    plt.show()

    # 场景2: 绘制纯白图, 像素值越接近0, 越黑, 越接近255, 越白.
    img2 = np.full(shape=(200, 200, 3), fill_value=255)
    plt.imshow(img2)
    # plt.axis('off') # 关闭坐标系, 就看不到白色的图了.
    plt.show()


# 2. 绘制 RGB真彩图
def dm02():
    # 1. 加载图片, 获取 像素矩阵.
    img1 = plt.imread('../../data/img.jpg')
    # print(f'img1: {img1}')
    # print(f'img1.shape: {img1.shape}')

    # 2. 保存图片, 其实就是保存: 像素点.
    # plt.imsave('./data/img_24.jpg', img1)

    # 3. 绘制图片.
    plt.imshow(img1)
    plt.show()

# 3. 测试
if __name__ == '__main__':
    # dm01()
    dm02()