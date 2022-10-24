import collections

import numpy as np
import PIL.Image
import torch
from torch.utils import data
import cv2
from torchvision import transforms
import os
from Toolkit import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CLASSES = ('Background', 'Building', 'Road',
           'Water', 'Barren', 'Forest', 'Agricultural')

PALETTE = np.array([[255, 255, 255], [255, 0, 0], [255, 255, 0],
                    [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]])


class LoveDaSeg(data.Dataset):

    def __init__(self, root, split=None, transform=True):
        self.root = root
        self.split = split
        self._transform = transform
        self.floders = ['Rural', 'Urban']

        self.files = collections.defaultdict(list)
        if self.split == 'Train' or self.split == 'Val':

            for floder in self.floders:
                img_dir = os.path.join(root, split, floder, 'images_png')
                img_files = os.listdir(img_dir)
                label_dir = os.path.join(root, split, floder, 'masks_png')
                label_files = os.listdir(label_dir)

                for img_file, label_file in zip(img_files, label_files):
                    self.files[split].append({'img': os.path.join(img_dir, img_file),
                                              'label': os.path.join(label_dir, label_file)})

        elif self.split == 'Test':
            for floder in self.floders:
                img_dir = os.path.join(root, split, floder, 'images_png')
                img_files = os.listdir(img_dir)

                for img_file in img_files:
                    self.files[split].append({'img': os.path.join(img_dir, img_file)})

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index, palette=PALETTE):
        data_file = self.files[self.split][index]
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        if self.split == 'Train' or self.split == 'Val':
            label_file = data_file['label']
            label = PIL.Image.open(label_file)
            label = np.array(label, dtype=np.uint8)

            img, label = transforms.randomFlip(img, label)
            img, label = transforms.randomCrop(img, label)
            img, label = transforms.resize(img, label, s=1024)

            fix_label = label.copy()
            label[fix_label >= 7] = 6
            label[fix_label == 6] = 5
            label[fix_label == 5] = 4
            label[fix_label == 4] = 3
            label[fix_label == 3] = 2
            label[fix_label == 2] = 1
            label[fix_label <= 1] = 0

            if self._transform:
                return transforms.transform(img, label)
            else:
                return img, label
        if self.split == 'Test':
            if self._transform:
                return transforms.predict_transform(img)

            else:
                return img


if __name__ == "__main__":
    batch_size = 1
    data_path = os.path.expanduser('D:/data/LoveDA/')

    test_data = LoveDaSeg(root=data_path, split='Train', transform=True)
    # (img, label) = test_data.__getitem__(index=35)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    for i, (img, label) in enumerate(test_loader):

        print('-----------------------------')
        print(torch.min(label))
        print(torch.max(label))
        print('-----------------------------')

        if i == 35:
            break
