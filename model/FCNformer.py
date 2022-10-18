from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from model import swin

class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class, backbone='swin-T'):
        super(FCN8s, self).__init__()

        expansion = 4 * 4 * 3 * 2
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        # H/32 * H/32 * 8C to H/16 * H/16 * 4C
        self.deconv1 = nn.ConvTranspose2d(8 * expansion, 4 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(4 * expansion)

        # H/16 * H/16 * 4C to H/8 * H/8 * 2C
        self.deconv2 = nn.ConvTranspose2d(4 * expansion, 2 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(2 * expansion)

        # H/8 * H/8 * 2C to H/4 * H/4 * C
        self.deconv3 = nn.ConvTranspose2d(2 * expansion, 1 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(1 * expansion)

        # H/4 * H/4 * C to H/2 * H/2 * C
        self.deconv4 = nn.ConvTranspose2d(1 * expansion, 1 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(1 * expansion)

        # H/2 * H/2 * C to H * H * C
        self.deconv5 = nn.ConvTranspose2d(1 * expansion, 1 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(1 * expansion)
        self.classifier = nn.Conv2d(1 * expansion, n_class, kernel_size=1)

        # 1 Initialize the parameters of the network
        # so that the variances of the output of each layer should be as equal as possible

        init.xavier_uniform_(self.deconv1.weight)  # Uniform distribution ~ U(−a,a)
        # 2
        init.xavier_uniform_(self.deconv2.weight)
        # 3
        init.xavier_uniform_(self.deconv3.weight)
        init.xavier_uniform_(self.deconv4.weight)
        init.xavier_uniform_(self.deconv5.weight)
        init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]
        score = self.relu(self.deconv1(x5))  # size=[n, 512, x.h/16, x.w/16] First deconvolution layer
        # print(score.shape, x4.shape)
        score = self.bn1(score + x4)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))  # size=[n, 256, x.h/8, x.w/8] second deconvolution layer
        score = self.bn2(score + x3)  # merged with the third maximum pooling layer
        score = self.relu(self.deconv3(score))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn3(score)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


if __name__ == '__main__':
    import torch

    input_size = 32*32

    backbone = swin.swin_t()
    model = FCN8s(pretrained_net=backbone, n_class=7, backbone='swin-T')
    # print(model)

    from torchsummary import summary

    summary(model, input_size=[(3, input_size, input_size)], device="cpu")