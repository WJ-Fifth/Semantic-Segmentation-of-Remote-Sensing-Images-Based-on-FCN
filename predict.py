import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.autograd import Variable
from Toolkit import tools


def main(args, model, test_data):
    use_cuda = torch.cuda.is_available()

    if args.model == 'fcn_resnet':
        model.load_state_dict(torch.load('./checkpoints/fcn_resnet_GID_checkpoint.pth')['state_dict'])

        print("The FCN ResNet model is defined and load checkpoint")

    elif args.model == 'fcn_vgg':
        # FCN8s Model Calling
        model.load_state_dict(torch.load('./checkpoints/fcn_vgg_checkpoint.pth'))
        print("The FCN VGG16 model is defined and load checkpoint")

    elif args.model == 'FCNformer':
        model.load_state_dict(torch.load('./checkpoints/FCNformer_GID_checkpoint.pth')['state_dict'])
        print("The FCNformer model is defined and load checkpoint")
    model.eval()

    if use_cuda:
        model.cuda()

    for i in range(len(test_data)):
        idx = i
        img = test_data[idx]

        img_name = str(i)

        if use_cuda:
            img = img.cuda()
        with torch.no_grad():
            img = Variable(img.unsqueeze(0))
        out = model(img)

        net_out = out.data.max(1)[1].squeeze_(0)
        if use_cuda:
            net_out = net_out.cpu()
        save_name = img_name + "_" + args.model
        tools.labelTopng(net_out, 'output/{}.png'.format(save_name))  # Converting web output to images
    print("over")
