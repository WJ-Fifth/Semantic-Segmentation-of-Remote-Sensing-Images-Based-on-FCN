from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet
from torch.nn import init
import numpy as np


def get_upsample_weight(in_channels, out_channels, kernel_size):  # 上采样权重提取
    '''
    make a 2D bilinear kernel suitable for upsampling
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]  # list (64 x 1), (1 x 64)
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)  # 64 x 64
    weight = np.zeros((in_channels, out_channels, kernel_size,
                       kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        score = self.bn1(self.relu(self.deconv1(x5)))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn2(self.relu(self.deconv2(score)))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn3(self.relu(self.deconv3(score)))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


class FCN16s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]

        score = self.relu(self.deconv1(x5))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x4)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.bn2(self.relu(self.deconv2(score)))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn3(self.relu(self.deconv3(score)))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

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
        score = self.bn1(score + x4)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))  # size=[n, 256, x.h/8, x.w/8] second deconvolution layer
        score = self.bn2(score + x3)  # merged with the third maximum pooling layer
        score = self.bn3(self.relu(self.deconv3(score)))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score

    # Initialization of the network as a whole:
    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                # if m.bias is not None:
                m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsample_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


ranges = {
        'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
        'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
        'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
        'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
        }

cfg = {'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
       }


def make_layers(cfg, batch_norm=False):  # normalizes BN layers to prevent gradient explosion and gradient dispersion

    layers = []
    in_channels = 3
    for layer_info in cfg:
        if layer_info == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            out_channels = layer_info
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels

    return nn.Sequential(*layers)


class VGGNet(VGG):
    def __init__(self, pretrained=True, requires_grad=False, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg['vgg16']))
        self.ranges = ranges['vgg16']

        if pretrained:
            vgg16 = models.vgg16(pretrained=False)
            vgg16.load_state_dict(torch.load('E:/FCN_Code/model/vgg16-397923af.pth'))  # Pre-training dataset
            # print("vgg16\n", vgg16)
            #
            # modified_features = nn.Sequential(*list(vgg16.features.children())[:-1])
            # # to relu5_3
            # print('modified_features:\n', modified_features)
            # exit()

        if not requires_grad:  # Fixed feature extraction layer
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # Removing the fully connected layer in VGG16
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits


# testing code
if __name__ == "__main__":
    net = VGGNet()
    # input_img = torch.ones((1, 3, 256, 256))
    # out = net(input_img)
    # print(net)
    print(FCN8s(net, n_class=6))
