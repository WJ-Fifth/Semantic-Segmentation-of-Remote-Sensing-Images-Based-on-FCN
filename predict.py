import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.autograd import Variable
from dataloaders import GID_loader, LoveDa_loader
from model import models, FCNformer, fcn_resnet
from Toolkit import tools

n_class = 6


def main():
    use_cuda = torch.cuda.is_available()
    path = os.path.expanduser('./data/LoveDA')

    # dataset = GID_loader.VOC2012ClassSeg(root=path, split='predict', transform=True)
    dataset = LoveDa_loader.LoveDaSeg(root=path, split='Test', transform=True)
    print("predict data load success")

    backbone = FCNformer.Swin_T()
    model = FCNformer.FCN8s(pretrained_net=backbone, n_class=6, backbone='swin-T')
    model.load_state_dict(torch.load('./checkpoints/FCNformer_GID_checkpoint.pth')['state_dict'])

    resnet = fcn_resnet.RESNET(pretrained=True, requires_grad=False)
    model = fcn_resnet.FCN8s(pretrained_net=resnet, n_class=7, backbone='resnet50')
    model.load_state_dict(torch.load('./checkpoints/fcn_resnet_GID_checkpoint.pth')['state_dict'])

    # vgg_model = models.VGGNet(pretrained=False, requires_grad=False)
    # fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)
    # fcn_model.load_state_dict(torch.load('model/model30.pth'))
    print("model load success")

    # fcn_model.eval()

    if use_cuda:
        model.cuda()

    # criterion = nn.CrossEntropyLoss()

    for i in range(len(dataset)):
        idx = i
        img = dataset[idx]

        img_name = str(i)

        if use_cuda:
            img = img.cuda()
        with torch.no_grad():
            img = Variable(img.unsqueeze(0))
        # print(img)
        out = model(img)
        # o = torch.argmax(out,dim=1)

        net_out = out.data.max(1)[1].squeeze_(0)
        # net_out = out.max(1)[1].squeeze()
        if use_cuda:
            net_out = net_out.cpu()

        tools.labelTopng(net_out, 'output/%s_fcnformer.png' % img_name)  # Converting web output to images
        # if i > 3:
        #     break
    print("over")


if __name__ == '__main__':
    main()
