import collections
import os.path as osp
import numpy as np
import PIL.Image
import torch
from torch.utils import data
import cv2
import random
from matplotlib import pyplot as plt


class VOCClassSegBase(data.Dataset):
    # 提供的标签分为6类，分别为建筑（标记1）、农场（标记2）、森林（标记3）、绿地（标记4）、水域（标记5）以及其他（标记0）

    class_names = np.array(['other', 'built_up', 'farmland', 'forest', 'meadow', 'water'])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])  # subtract mean  减去均值

    def __init__(self, root, split='train', transform=True):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = osp.join(self.root)  # 合并VOC2012数据集路径
        # dataset_dir = osp.join(self.root, 'Large-scale Classification_5classes/')  # 合并VOC2012数据集路径
        self.files = collections.defaultdict(list)
        if self.split == 'train':
            # for split_file in ['train', 'val']:
            for split_file in ['train']:
                imgsets_file = osp.join(dataset_dir, '%s.txt' % split_file)  # ImageSets train.txt
                for img_name in open(imgsets_file):
                    img_name = img_name.strip()
                    img_file = osp.join(dataset_dir, 'img', img_name)  # 提供的是VOC所提供的所有的图片信息，包括训练图片
                    lbl_file = img_file.replace('img', 'label')
                    self.files[split_file].append({'img': img_file, 'lbl': lbl_file})

        if self.split == 'test':
            # for split_file in ['test', 'test_val']:
            for split_file in ['test']:
                imgsets_file = osp.join(dataset_dir, '%s.txt' % split_file)  # ImageSets test.txt
                for img_name in open(imgsets_file):
                    img_name = img_name.strip()
                    img_file = osp.join(dataset_dir, 'img', img_name)  # 提供的是VOC所提供的所有的图片信息，包括训练图片
                    lbl_file = img_file.replace('img', 'label')
                    self.files[split_file].append({'img': img_file, 'lbl': lbl_file})

        if self.split == 'predict':
            for split_file in ['predict']:
                imgsets_file = osp.join(dataset_dir, '%s.txt' % split_file)
                for img_name in open(imgsets_file):
                    img_name = img_name.strip()
                    img_file = osp.join(dataset_dir, 'img_predict', img_name)
                    self.files[split_file].append({'img': img_file})

    # 返回数据集长度
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):  # 迭代器

        # print('getitem方法被调用')
        data_file = self.files[self.split][index]  # 数据
        # load image

        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        # print(img_file)
        # print(img_file)
        # img = cv2.imread(img_file)
        img = np.array(img, dtype=np.uint8)
        if self.split == 'train' or self.split == 'test':
            # load label
            lbl_file = data_file['lbl']
            # lbl_file = "data/label/GF2_PMS1__L1A0000564539-MSS1.tif_0_3200.png"
            # print(lbl_file)
            lbl = PIL.Image.open(lbl_file)
            lbl = np.array(lbl, dtype=np.uint8)
            # lbl = lbl[:, :, 0]
            # rgb标签转化为0,1,2,3,4,5
            # plt.imshow(lbl)
            # plt.show()
            lbl = lbl // 255 * [[[4, 2, 1]]]
            lbl = np.sum(lbl, axis=2)
            lbl1 = lbl.copy()
            # print(np.unique(lbl1))
            lbl[lbl1 == 7] = 0
            lbl[lbl1 == 4] = 1
            lbl[lbl1 == 2] = 2
            lbl[lbl1 == 3] = 3
            lbl[lbl1 == 6] = 4
            lbl[lbl1 == 1] = 5

            img, lbl = self.randomFlip(img, lbl)
            img, lbl = self.randomCrop(img, lbl)
            img, lbl = self.resize(img, lbl)
            # label = np.asarray(lbl)
            # print(np.unique(label))
            if self._transform:
                return self.transform(img, lbl)
            else:
                return img, lbl
        if self.split == 'predict':
            if self._transform:
                return self.predict_transform(img)

            else:
                return img

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        # img -= self.mean_bgr
        img = img / 255.
        img = img.transpose(2, 0, 1)  # whc -> cwh 表示矩阵XYZ轴发生变换
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def predict_transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        # img -= self.mean_bgr
        img = img / 255.
        img = img.transpose(2, 0, 1)  # whc -> cwh
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)  # cwh -> whc
        # img += self.mean_bgr
        img = img * 255
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]  # BGR -> RGB
        return img

    def randomFlip(self, img, label):  # 随机反转
        if random.random() < 0.5:  # 伪随机50%的概率进行反转
            img = np.fliplr(img)  # 实现numpy数组的随机反转
            label = np.fliplr(label)
        return img, label

    def predict_randomFlip(self, img):
        if random.random() < 0.5:
            img = np.fliplr(img)
        return img

    def resize(self, img, label, s=640):  # s=640
        # print(s, img.shape)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (s, s), interpolation=cv2.INTER_NEAREST)
        return img, label

    def predict_resize(self, img, s=640):  # s=640
        # print(s, img.shape)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        return img

    def randomCrop(self, img, label):  # 随机裁剪
        h, w, _ = img.shape  # 读取图片高度和宽度信息
        short_size = min(w, h)  # 取高度和宽度中的较小值
        rand_size = random.randrange(int(0.7 * short_size), short_size)
        x = random.randrange(0, w - rand_size)
        y = random.randrange(0, h - rand_size)

        return img[y:y + rand_size, x:x + rand_size], label[y:y + rand_size, x:x + rand_size]

    def predict_randomCrop(self, img):  # 随机裁剪预测图像
        h, w, _ = img.shape
        short_size = min(w, h)
        rand_size = random.randrange(int(0.7 * short_size), short_size)
        x = random.randrange(0, w - rand_size)
        y = random.randrange(0, h - rand_size)
        return img[y:y + rand_size, x:x + rand_size]

    # data augmentation
    def augmentation(self, img, lbl):
        img, lbl = self.randomFlip(img, lbl)
        img, lbl = self.randomCrop(img, lbl)
        img, lbl = self.resize(img, lbl)
        return img, lbl


class VOC2012ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=True):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform)
