from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet
from torch.nn import init
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_upsample_weight(in_channels, out_channels, kernel_size):  # Up-sampling weight extraction
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
    def __init__(self, pretrained_net, n_class, backbone='resnet18'):
        super(FCN8s, self).__init__()

        if backbone == 'resnet18' or backbone == 'resnet34':
            expansion = 1
        elif backbone == 'resnet50' or backbone == 'resnet101':
            expansion = 4
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512 * expansion, 256 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256 * expansion)
        self.deconv2 = nn.ConvTranspose2d(256 * expansion, 128 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128 * expansion)
        self.deconv3 = nn.ConvTranspose2d(128 * expansion, 64 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64 * expansion)
        self.deconv4 = nn.ConvTranspose2d(64 * expansion, 64 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64 * expansion)
        self.deconv5 = nn.ConvTranspose2d(64 * expansion, 32 * expansion,
                                          kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32 * expansion)
        self.classifier = nn.Conv2d(32 * expansion, n_class, kernel_size=1)

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


cfg_resnet = {'resnet18': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']}


class RESNET(nn.Module):

    def __init__(self, pretrained=False, requires_grad=False, remove_fc=True, show_params=False):
        super().__init__()
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.backbone = models.resnet50()

        if not requires_grad:  # Fixed feature extraction layer
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # Removing the fully connected layer in ResNet
            del self.backbone.fc

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)

        output["x%d" % 2] = x

        x = self.backbone.layer2(x)
        output["x%d" % 3] = x
        x = self.backbone.layer3(x)
        output["x%d" % 4] = x
        x = self.backbone.layer4(x)
        output["x%d" % 5] = x
        return output


# testing code
if __name__ == "__main__":
    from torchsummary import summary

    resnet = RESNET()
    # print(resnet)

    # summary(resnet, input_size=[(3, 640, 640)], device="cpu")
    FCN = FCN8s(pretrained_net=resnet, n_class=7, backbone='resnet50')
    # print(FCN)
    summary(FCN, input_size=[(3, 640, 640)], device="cpu")
