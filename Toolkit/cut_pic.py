from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2


def fen_ge(img_path, save_txt_path):
    img_list = glob(img_path)
    print(img_list)
    f = open(save_txt_path, 'w')
    width = 1024
    height = 1024
    for img_path in img_list:
        print(img_path)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
        print(img.shape)

        for i in range(0, img.shape[0], width):
            for j in range(0, img.shape[1], height):
                if (i + width) > img.shape[0] or (j + height) > img.shape[1]:
                    continue
                save_img = img[i:min(i + width, img.shape[0]), j:min(j + height, img.shape[1]), :]
                if 'label' in img_path:
                    save_path = os.path.join('../data/GID-5/label', os.path.basename(img_path))
                else:
                    save_path = os.path.join('../data/GID-5/img', os.path.basename(img_path))

                save_path = save_path.replace('_label', '')

                save_path = save_path + "_" + str(i) + "_" + str(j) + ".png"
                if '_label' not in img_path:
                    print(os.path.basename(save_path), file=f)
                save_img = Image.fromarray(np.uint8(save_img))
                save_img.save(save_path)
    print(img_list)


def fen_ge_predict(img_path, save_txt_path):
    img_list = glob(img_path)
    f = open(save_txt_path, 'w')
    width = 1024
    height = 1024
    for img_path in img_list:
        print(img_path)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
        print(img.shape)
        # img = Image.open(img_path)
        # img = np.array(img)
        # print(img)
        for i in range(0, img.shape[0], width):
            for j in range(0, img.shape[1], height):
                if (i + width) > img.shape[0] or (j + height) > img.shape[1]:
                    continue
                save_img = img[i:min(i + width, img.shape[0]), j:min(j + height, img.shape[1]), :]

                save_path = os.path.join('../data/GID-5/img_predict', os.path.basename(img_path))

                save_path = save_path + "_" + str(i) + "_" + str(j) + ".png"
                if '_label' not in img_path:
                    print(os.path.basename(save_path), file=f)
                save_img = Image.fromarray(np.uint8(save_img))
                save_img.save(save_path)
    print(img_list)


if not os.path.exists('../data/GID-5/img/'):
    os.makedirs('../data/GID-5/img')
if not os.path.exists('../data/GID-5/label/'):
    os.makedirs('../data/GID-5/label')


if __name__ == '__main__':
    fen_ge('../data/GID-5/train_ori/*.tif', '../data/GID-5/train.txt')
    fen_ge('../data/GID-5/val_ori/*.tif', '../data/GID-5/val.txt')
    # fen_ge('./data/test_val_ori/*.tif', './data/test_val.txt')
    # fen_ge_predict('data/predict_ori/*.tif', 'data/predict.txt')
    print("finish")
