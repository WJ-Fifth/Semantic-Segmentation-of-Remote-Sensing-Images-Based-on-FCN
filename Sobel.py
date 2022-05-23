import cv2
import numpy as np
from PIL import Image
import math


def Sobel(path, save_path, number):
    path = path + "%d_src.png" % number
    img = cv2.imread(path)

    Gx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    Gy = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(Gx)  # Convert data to 8-bit uint8 by linear transformation
    absY = cv2.convertScaleAbs(Gy)

    # Tend = math.atan2(absY, absX)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # Fusion of two images
    save_path = save_path + "%d_edge.png" % number
    cv2.imwrite(save_path, dst)

    edge = cv2.Canny(dst, 50, 100)
    # cv2.imshow("Canny_edge_1", edge)
    edge1 = cv2.Canny(Gx, Gy, 10, 100)
    # cv2.imshow("Canny_edge_2", edge1)
    # Use edges as masks for bitwise_and bitwise operations
    edge2 = cv2.bitwise_and(img, img, mask=edge1)
    # cv2.imshow("bitwise_and", edge2)
    save_path1 = save_path + "%d_edge_canny.png" % number
    cv2.imwrite(save_path1, edge2)


for i in range(30):
    # path = "D:/VOC2012/JPEGImages/"
    # save_path = "D:/VOC2012/JPEGImages/EdgeDetection/"
    path = "E:/FCN_Code/fcn_pytorch/data/result/"
    save_path = "E:/FCN_Code/fcn_pytorch/data/result/edge/"
    Sobel(path, save_path, i)
print("Edge extraction completed")
