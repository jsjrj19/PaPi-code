# Modified from https://github.com/hbzju/PiCO/blob/main/utils/cifar10.py

import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from collections import OrderedDict

from .wide_resnet import WideResNet
from .utils_algo import generate_uniform_cv_candidate_labels, generate_instancedependent_candidate_labels
from .cutout import Cutout
from .autoaugment import CIFAR10Policy, ImageNetPolicy


def load_cifar10(args):
    #测试集的图像变换
    test_transform = transforms.Compose([
            transforms.ToTensor(),#转换为pytorch张量
            #标准化
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    #加载原始cifar10数据集
    original_train = dsets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    #提取原始图像数据和标签
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()

    #加载测试集并应用测试变换
    test_dataset = dsets.CIFAR10(root=args.data_dir, train=False, transform=test_transform)
    #创建测试集数据加载器
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers, \
                                              sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    #生成不同的部分标签矩阵
    if args.exp_type == 'rand':
        partialY_matrix = generate_uniform_cv_candidate_labels(args, ori_labels)
    elif args.exp_type == 'ins':#实例依赖的部分标签
        #原始数据转换为Tensor
        ori_data = torch.Tensor(original_train.data)
        #定义wideresnet模型用于预测特征，生成实例依赖标签
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        #加载预训练模型权重
        model.load_state_dict(torch.load('./pmodel/cifar10.pt'))
        #调整图像维度顺序
        ori_data = ori_data.permute(0, 3, 1, 2)
        #生成实例依赖的部分标签矩阵
        partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels)
        #再将ori_data还原为原始numpy格式
        ori_data = original_train.data

    #创建真实标签的one-hot矩阵
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1#真实标签位置设为1

    #检查每个样本的部分标签是否包含真实标签
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print('Inconsistent permutation !')
    #打印平均每个样本的候选标签数量
    print('Average candidate num: ', partialY_matrix.sum(1).mean())
    #构建带部分标签的训练数据集
    partial_training_dataset = CIFAR10_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())
    #创建分布式训练采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_training_dataset)
    #创建训练集数据加器
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,#多进程加载数据的进程数
        pin_memory=True,#锁存内存
        sampler=train_sampler,
        drop_last=True
    )
    
    return partial_training_dataloader, partialY_matrix, train_sampler, test_loader


class CIFAR10_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        #存储原始图像
        self.ori_images = images
        #存储部分标签矩阵
        self.given_partial_label_matrix = given_partial_label_matrix
        #存储真实标签
        self.true_labels = true_labels
        #定义第一种图像变换：弱增强
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        #定义第二种图像变换：强增强
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __len__(self):
        #返回数据集样本长度
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        each_image1 = self.transform1(self.ori_images[index])
        each_image2 = self.transform2(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image1, each_image2, each_label, each_true_label, index


