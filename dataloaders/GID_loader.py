import collections
import os.path
import numpy as np
import PIL.Image
import torch
from torch.utils import data
import cv2
import random
from matplotlib import pyplot as plt
import os
from Toolkit import transforms


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CLASSES = ('other', 'built_up', 'farmland', 'forest', 'meadow', 'water')

PALETTE = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0],
                    [0, 255, 255], [255, 255, 0], [0, 0, 255]])


class VOCClassSegBase(data.Dataset):
    """
    The tags provided are divided into six categories:
    buildings (tag 1), farms (tag 2), forests (tag 3), green spaces (tag 4), waters (tag 5), and others (tag 0)
    """

    class_names = np.array(['other', 'built_up', 'farmland', 'forest', 'meadow', 'water'])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])  # subtract mean

    def __init__(self, root, split='train', transform=True):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = os.path.join(self.root)  # Merge VOC2012 dataloaders paths
        # dataset_dir = osp.join(self.root, 'Large-scale Classification_5classes/')
        self.files = collections.defaultdict(list)
        if self.split == 'train':
            # for split_file in ['train', 'val']:
            for split_file in ['train']:
                imgsets_file = os.path.join(dataset_dir, '%s.txt' % split_file)  # ImageSets train.txt
                for img_name in open(imgsets_file):
                    img_name = img_name.strip()
                    img_file = os.path.join(dataset_dir, 'img', img_name)
                    # Provided is all the image information provided by the VOC, including training images

                    label_file = img_file.replace('img', 'label')
                    self.files[split_file].append({'img': img_file, 'label': label_file})

        if self.split == 'test':
            # for split_file in ['test', 'test_val']:
            for split_file in ['test']:
                imgsets_file = os.path.join(dataset_dir, '%s.txt' % split_file)  # ImageSets test.txt
                for img_name in open(imgsets_file):
                    img_name = img_name.strip()
                    img_file = os.path.join(dataset_dir, 'img', img_name)
                    # Provided is all the image information provided by the VOC, including training images

                    label_file = img_file.replace('img', 'label')
                    self.files[split_file].append({'img': img_file, 'label': label_file})

        if self.split == 'predict':
            for split_file in ['predict']:
                imgsets_file = os.path.join(dataset_dir, '%s.txt' % split_file)
                for img_name in open(imgsets_file):
                    img_name = img_name.strip()
                    img_file = os.path.join(dataset_dir, 'img_predict', img_name)
                    self.files[split_file].append({'img': img_file})

    # 返回数据集长度
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index, palette=PALETTE):  # Iterators

        data_file = self.files[self.split][index]
        # load image

        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        # print(img_file)
        # print(img_file)
        # img = cv2.imread(img_file)
        img = np.array(img, dtype=np.uint8)
        if self.split == 'train' or self.split == 'test':
            # load label
            label_file = data_file['label']
            # label_file = "data/label/GF2_PMS1__L1A0000564539-MSS1.tif_0_3200.png"
            # print(label_file)
            label = PIL.Image.open(label_file)
            label = np.array(label, dtype=np.uint8)
            label = label // 255 * [[[4, 2, 1]]]
            label = np.sum(label, axis=2, dtype=np.uint8)
            palette = np.sum((palette // 255 * [[[4, 2, 1]]]), axis=2, dtype=np.uint8).reshape(-1)
            label_index = label.copy()
            label[label_index == palette[0]] = 0  # [0, 0, 0]
            label[label_index == palette[1]] = 1  # [255, 0, 0]
            label[label_index == palette[2]] = 2  # [0, 255, 0]
            label[label_index == palette[3]] = 3  # [0, 255, 255]
            label[label_index == palette[4]] = 4  # [255, 255, 0]
            label[label_index == palette[5]] = 5  # [0, 0, 255]

            img, label = transforms.randomFlip(img, label)
            img, label = transforms.randomCrop(img, label)
            img, label = transforms.resize(img, label)
            print(np.max(label))
            print(np.min(label))
            if self._transform:
                return transforms.transform(img, label)
            else:
                return img, label
        if self.split == 'predict':
            if self._transform:
                return transforms.predict_transform(img)

            else:
                return img


class VOC2012ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=True):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform)


if __name__ == "__main__":
    label = PIL.Image.open('../data/label/GF2_PMS2__L1A0001577567-MSS2.tif_3200_1920.png')

    batch_size = 4
    data_path = os.path.expanduser('../data/')

    test_data = VOC2012ClassSeg(root=data_path, split='test', transform=True)
    test_data.__getitem__(index=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
