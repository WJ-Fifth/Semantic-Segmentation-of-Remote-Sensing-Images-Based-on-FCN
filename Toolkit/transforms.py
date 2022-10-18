import collections
import os.path as osp
import numpy as np
import PIL.Image
import torch
from torch.utils import data
import cv2
import random


def transform(img, label):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    # img -= self.mean_bgr
    img = img / 255.
    img = img.transpose(2, 0, 1)  # whc -> cwh indicates that the matrix XYZ axis is transformed
    img = torch.from_numpy(img.copy()).float()

    label = torch.from_numpy(label.copy()).long()
    return img, label


def predict_transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    # img -= self.mean_bgr
    img = img / 255.
    img = img.transpose(2, 0, 1)  # whc -> cwh
    img = torch.from_numpy(img).float()
    return img


def untransform(img):
    img = img.numpy()
    img = img.transpose(1, 2, 0)  # cwh -> whc
    # img += self.mean_bgr
    img = img * 255
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]  # BGR -> RGB
    return img


def randomFlip(img, label):  # Random inversion
    if random.random() < 0.5:  # Pseudo-random 50% probability of inversion
        img = np.fliplr(img)  # Implement random inversion of numpy arrays
        label = np.fliplr(label)
    return img, label


def predict_randomFlip(img):
    if random.random() < 0.5:
        img = np.fliplr(img)
    return img


def resize(img, label, s=640):  # s=640
    # print(s, img.shape)
    img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (s, s), interpolation=cv2.INTER_NEAREST)
    return img, label


def predict_resize(img, s=640):  # s=640
    # print(s, img.shape)
    img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    return img


def randomCrop(img, label):
    """
    Random cropping
    """
    h, w, _ = img.shape
    short_size = min(w, h)  # Take the smaller of the height and width
    rand_size = random.randrange(int(0.7 * short_size), short_size)
    x = random.randrange(0, w - rand_size)
    y = random.randrange(0, h - rand_size)

    return img[y:y + rand_size, x:x + rand_size], label[y:y + rand_size, x:x + rand_size]


def predict_randomCrop(img):
    h, w, _ = img.shape
    short_size = min(w, h)
    rand_size = random.randrange(int(0.7 * short_size), short_size)
    x = random.randrange(0, w - rand_size)
    y = random.randrange(0, h - rand_size)
    return img[y:y + rand_size, x:x + rand_size]


def augmentation(img, label):
    img, label = randomFlip(img, label)
    img, label = randomCrop(img, label)
    img, label = resize(img, label)
    return img, label


