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

    dataset = voc_loader.VOC2012ClassSeg(root=path, split='test', transform=True)
    print("predict data load success")

    vgg_model = models.VGGNet(pretrained=False,requires_grad=False)
    fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)
    fcn_model.load_state_dict(torch.load('model/model30.pth'))
    print("model load success")

    # fcn_model.eval()

    if use_cuda:
        fcn_model.cuda()

    # criterion = nn.CrossEntropyLoss()

    for i in range(len(dataset)):
        idx = i
        img = dataset[idx]
        img, label = dataset[idx]
        # print(np.unique(label))
        # exit(-1)
        img_name = str(i)
        # img_src = dataset.untransform(img)  # whc
        # plt.imshow(img_src)
        # plt.show()
        # cv2.imwrite(path + 'result/%s_src.png' % img_name, img_src)
        # cv2.imwrite(path + 'result/%s_src.png' % img_name, img)

        if use_cuda:
            img = img.cuda()

        img = Variable(img.unsqueeze(0), volatile=True)
        # print(img)
        out = fcn_model(img)
        o = torch.argmax(out,dim=1)
        # print(o)
        # exit(-1)
        net_out = torch.argmax(out,dim=1)
        o = net_out[0].cpu().numpy()
        label_img = np.concatenate([label.numpy(), o], axis=1)
        plt.imshow(label_img)
        plt.savefig(path + 'output_1/%s_out_p.png' % img_name)
        # plt.show()

        # print(np.unique(o))
        net_out = out.data.max(1)[1].squeeze_(0)
        # net_out = out.max(1)[1].squeeze()
        if use_cuda:
            net_out = net_out.cpu()

        tools.labelTopng(net_out, path + 'output_1/%s_out.png' % img_name)  # Converting output to images
        # if i > 3:
        #     break
    print("over")


if __name__ == '__main__':
    main()
