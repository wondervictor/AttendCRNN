#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import six
import pickle
import scipy.io as scio
from PIL import Image
import numpy as np


class IIIT5k(Dataset):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False
    """
    def __init__(self, root, training=True, fix_width=False):
        super(IIIT5k, self).__init__()
        data_str = 'traindata' if training else 'testdata'
        self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in
            scio.loadmat(os.path.join(root, data_str+'.mat'))[data_str][0]])

        # transform = [transforms.Resize((100, 32))]
        # transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        # transform = transforms.Compose(transform)

        def open_img(name):
            img = Image.open(root+'/'+name)
            copy_image = img.copy()
            img.close()
            return copy_image
        self.img = map(open_img, self.img)  # [Image.open(root+'/'+img) for img in self.img] # [transform(Image.open(root+'/'+img)) for img in self.img]
        print("Load Dataset: {} Finished".format('train' if training else 'test'))

    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx].astype(str)


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.gray = transforms.Grayscale()
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.gray(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        super(RandomSequentialSampler, self).__init__(data_source)
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
            # deal with tail
            if tail:
                random_start = random.randint(0, len(self) - self.batch_size)
                tail_index = random_start + torch.range(0, tail - 1)
                index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):

    def __init__(self, img_height=32, img_weight=100, keep_ratio=False, min_ratio=1):
        self.img_height = img_height
        self.img_weight = img_weight
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        img_height = self.img_height
        img_weight = self.img_weight
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            img_weight = int(np.floor(max_ratio * img_height))
            img_weight = max(img_height * self.min_ratio, img_weight)  # assure imgH >= imgW

        transform = ResizeNormalize((img_weight, img_height))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

