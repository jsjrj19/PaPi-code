# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py

import numpy as np

import torch


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes#存储要遮挡的区域数量
        self.length = length#存储每个遮挡区域的边长

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        #获取图像的高度和宽度
        h = img.size(1)
        w = img.size(2)

        #创建一个全为一的掩码矩阵
        mask = np.ones((h, w), np.float32)

        #循环生成n_holes个遮挡区域
        for n in range(self.n_holes):
            #随机选择遮挡区域的中心点坐标
            y = np.random.randint(h)
            x = np.random.randint(w)

            #计算遮挡区域的左上角和右下角坐标（确保不超出图像边界）
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            #将掩码矩阵中遮挡区域的位置设为0
            mask[y1: y2, x1: x2] = 0.
        #将numpy掩码矩阵转换为pytorch张量
        mask = torch.from_numpy(mask)
        #将掩码扩展为与输入图像相同的形状
        mask = mask.expand_as(img)
        #图像与掩码相乘
        img = img * mask

        return img

