# https://github.com/hbzju/PiCO/blob/main/utils/utils_algo.py

import math
import pickle
import numpy as np
import copy
from scipy.special import comb

import torch
import torch.nn as nn
import torch.nn.functional as F



def generate_instancedependent_candidate_labels(model, train_X, train_Y):
    #禁用梯度计算：仅用于模型预测，不训练
    with torch.no_grad():

        #计算类别数量k和样本数量n
        k = int(torch.max(train_Y) - torch.min(train_Y) + 1)
        n = train_Y.shape[0]
        #将模型移动到GPU
        model = model.cuda()
        #将真实标签转换为one-hot编码
        train_Y = torch.nn.functional.one_hot(train_Y, num_classes=k)

        #初始化变量：平均候选标签数，存储各批次部分标签的列表
        avg_C = 0
        partialY_list = []
        #初始化额外标签比例和批处理大小
        rate, batch_size = 0.4, 2000
        #计算批次数
        step = math.ceil(n / batch_size)
        
        print('Instance dependent partializing...')
        #按批次处理训练数据
        for i in range(0, step):
            #计算当前批次的结束索引
            b_end = min((i + 1) * batch_size, n)

            #提取当前批次的特征数据
            train_X_part = train_X[i * batch_size : b_end].cuda()
            #预测当前批次的输出
            outputs = model(train_X_part)
            #提取当前批次的one_hot标签，创建副本并分离计算图
            train_p_Y = train_Y[i * batch_size : b_end].clone().detach()

            #######计算额外标签生成概率#########
            #对模型输出做softmax，得到类别概率分布
            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            #真实标签对应的概率设为0
            partial_rate_array[torch.where(train_p_Y == 1)] = 0
            #行归一化
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            #调整概率均值：使每行平均概率等于rate
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            #截断概率
            partial_rate_array[partial_rate_array > 1.0] = 1.0

            ####按概率采样额外标签####
            #创建二项分布对象
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            #采样结果
            z = m.sample()
            #将采样的额外标签添加到原始one_hot标签中
            train_p_Y[torch.where(z == 1)] = 1.0
            #将当前批次的部分标签添加到列表
            partialY_list.append(train_p_Y)

        #拼接所有批次的部分标签
        partialY = torch.cat(partialY_list, dim=0).float()
        #确保生成部分标签数量与原始样本数量一致
        assert partialY.shape[0] == train_X.shape[0]
    #计算平均每个样本的候选标签数    
    avg_C = torch.sum(partialY) / partialY.size(0)
    
    print('avg_C: ', avg_C)
    
    return partialY



def generate_uniform_cv_candidate_labels(args, train_labels):
    #异常检查：若标签最小值大于1说明标签格式错误
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    #标签格式转换
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1
    #计算类别数和样本数
    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    #初始化部分标签矩阵
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    #设置非真实标签成为候选的概率
    p_1 = args.partial_rate
    #构建转移矩阵
    transition_matrix = np.eye(K)#对角线真实标签为1
    #非对角线为p_1
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = p_1
    print('Transition_matrix ==>')
    print(transition_matrix)

    #生成随机数矩阵
    random_n = np.random.uniform(0, 1, size=(n, K))

    #遍历每个样本和每个类别，生成部分标签
    for j in range(n):#j：样本索引
        for jj in range(K):#类别索引
            if jj == train_labels[j]:#跳过真实标签
                continue
            #若随机数小于转移矩阵概率，则将该类别设为候选标签
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Uniform Candidate Label Sets!\n")
    
    return partialY



def generate_hierarchical_cv_candidate_labels(dataname, train_labels, args):
    assert dataname == 'cifar100'

    meta = unpickle('../data/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    
    hierarchical = {}
    reverse_hierarchical = {}
    
    hierarchical_idx = [None] * 20
    
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    
    # class to superclass
    super_classes = []
    labels_by_h = []
    
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    
    p_1 = args.partial_rate
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = p_1
    
    mask = np.zeros_like(transition_matrix)
    
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print('transition_matrix')
    print(transition_matrix)
    print()
    
    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    
    print("Finish Generating Hierarchical Candidate Label Sets!\n")
    
    return partialY


#读取pickle格式文件
def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res


#统计指标平均值
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        #初始化
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        #充置当前值
        self.val = 0#当前批次的指标值
        self.avg = 0#累计平均值
        self.sum = 0#累计总和
        self.count = 0#累计样本数

    def update(self, val, n=1):#更新统计
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):#定义打印格式
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

#打印训练进度
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    #打印当前批次进度
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    #计算 批次数字的位数
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
#动态调整学习率
def adjust_learning_rate(args, optimizer, epoch):
    #初始学习率
    lr = args.lr
    #余弦退火学习率
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        #阶梯式学习率
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    
#计算top-k精度
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        #获取topk的预测结果
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #比较预测与真实标签
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        #计算每个k对应的精度
        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res
    

#计算数据集整体精度
def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0#正确样本数，总样本数
        #遍历数据加载器的每个批次
        for images, labels in loader:
            #将标签和图像移动到指定设备
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images)
            #获取预测类别
            _, predicted = torch.max(outputs.data, 1)
            #累加正确预测类别
            total += (predicted == labels).sum().item()
            #累加总样本数
            num_samples += labels.size(0)
    #返回总体精度
    return total / num_samples


#S型递增函数
def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    #若rampup长度为0，直接返回1.0
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))

    
#线性递增函数，用于动态调整损失权重
def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

#余弦递减函数：用于动态调整损失权重   
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


