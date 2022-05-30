import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
import torch.nn as nn
from torch.autograd import Variable
import voc_loader
import models
import tools
from matplotlib import pyplot as plt
import numpy as np

n_class = 6


def main():
    use_cuda = torch.cuda.is_available()
    path = os.path.expanduser('data/')

    dataset = voc_loader.VOC2012ClassSeg(root=path, split='predict', transform=True)
    print("predict data load success")

    vgg_model = models.VGGNet(pretrained=False, requires_grad=False)
    fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)
    fcn_model.load_state_dict(torch.load('./model/model100.pth'))
    print("model load success")

    # fcn_model.eval()

    if use_cuda:
        fcn_model.cuda()

    # criterion = nn.CrossEntropyLoss()

    for i in range(len(dataset)):
        idx = i
        img = dataset[idx]

        img_name = str(i)

        if use_cuda:
            img = img.cuda()

        with torch.no_grad():
            img = Variable(img.unsqueeze(0), volatile=True)

        out = fcn_model(img)
        # o = torch.argmax(out,dim=1)

        net_out = out.data.max(1)[1].squeeze_(0)
        # net_out = out.max(1)[1].squeeze()
        if use_cuda:
            net_out = net_out.cpu()

        tools.labelTopng(net_out, path + 'output/%s_out.png' % img_name)  # Converting web output to images
        # if i > 3:
        #     break
    print("over")


if __name__ == '__main__':
    main()
