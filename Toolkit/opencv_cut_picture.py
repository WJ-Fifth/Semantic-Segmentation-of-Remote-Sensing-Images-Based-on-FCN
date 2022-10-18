import cv2
import numpy as np

# Read in images from a folder
def readpicture(path, number, type):
    my_path = path + str(number) + str(type)
    print(my_path)
    img = cv2.imread(my_path)  # set the format of the image to be read in
    return img


def ROIselect(img, size):
    x_num = img.shape[0] / size
    y_num = img.shape[1] / size
    x = img.shape[0]  # Vertical dimensions (height) of the image
    y = img.shape[1]  # Horizontal dimensions (width) of the image
    print(x)
    print(y)
    image_zoom = np.zeros(shape=(size, size, 3, int(x_num * y_num)), dtype='uint8')
    order = 0  # Segmented image number
    for i in range(size, x + size, size):
        for k in range(size, y + size, size):
            img_new = img[i - size:i, k - size:k]
            image_zoom[:, :, :, order] = img_new
            order = order + 1
    return image_zoom


def montage_ROIselect(img, image_zoom, x, y, size):
    k = x % y
    y = x // y
    image_zoom[int(y * size):int((y + 1) * size), int(k * size):int((k + 1) * size)] = img
    return image_zoom


def size_change(img, size):
    x_size = (size - img.shape[0] % size) + img.shape[0]
    y_size = (size - img.shape[1] % size) + img.shape[1]
    img = cv2.resize(img, (x_size, y_size))
    return img


def save_picture(img, path_save, savenumber, mytype):
    path_save = path_save + str(savenumber) + str(mytype)
    cv2.imwrite(path_save, img)
    print("Successful storage of images%d", savenumber)


def cut_picture(read_path, save_path, size):
    type_mask = ".png"
    number = 1
    save_path_number = 1
    while number <= 150:
        img = readpicture(read_path, number, type_mask)

        image = size_change(img, size)
        imagezoom = ROIselect(image, size)
        order_new = imagezoom.shape[3]  # Number of images, fourth element of shape
        for arrangement in range(0, order_new):
            save_picture(imagezoom[:, :, :, arrangement], save_path, save_path_number, type_mask)
            save_path_number = save_path_number + 1
        number = number + 1
        print("After traversal")
        print("A total of split" + str(number-1) + "images")


# Image Mosaic of prediction results
def montage_picture(read_path, save_path, s1, s2, s3, start, end, s, s5, s6):
    mytype = "_out.png"
    type_mask = "pre.png"
    number = start
    x = 0
    save_path_number = 1
    image_zoom = np.zeros(shape=(s1, s2, s3), dtype='uint8')
    print(image_zoom.shape)
    while number <= end:
        img = readpicture(read_path, number, mytype)
        image_zoom = montage_ROIselect(img, image_zoom, x, s2 / s, s)
        x = x + 1
        number = number + 1
    print("After traversal")
    img = cv2.resize(image_zoom, (s5, s6))
    cv2.imwrite(save_path + type_mask, img)
    # cv2.imshow("label_image", img)

'''
The parameters are, in order, read image address, segmentation result storage address, segmentation size

The parameters are, in order, the address to read the image, 
the address to store the stitching result, the size of the image (3), 
the stitching start position, the stitching end position, the size of the small image, 
and the size of the original image (2).
'''
if __name__ == '__main__':

    # image data path
    path = "E:/FCN_Code/GID5/val_data/"
    save_path = "E:/FCN_Code/GID-5/img/"

    # cut_picture: Implementing segmentation of dataset images
    cut_picture(path, save_path, 1024)

    # montage_picture(path, save_path, 7680, 7040, 3, 0, 109, 640, 7300, 6908)
    # montage_picture(path, save_path, 7680, 7040, 3, 0, 109, 640, 7200, 6800)


