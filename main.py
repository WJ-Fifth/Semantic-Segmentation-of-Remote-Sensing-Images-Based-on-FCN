import argparse
import torch
from dataloaders import LoveDa_loader, GID_loader
from model import fcn_resnet, models, swin, FCNformer
from train import train, resume, evaluate
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='exp_fcn_resnet')

    parser.add_argument('--model', type=str, default='fcn_vgg',
                        help='with/without using dynamic filters')

    parser.add_argument('--dataset', type=str, default='LoveDa',
                        help='select dataloaders with GID, LoveDa etc.')

    parser.add_argument('--data_path', type=str, default='D:/data/LoveDA',
                        help='select dataloaders path with GID, LoveDa etc.')

    parser.add_argument('--resume', type=int, default=0,
                        help='resume the trained model')
    parser.add_argument('--test', type=int, default=0,
                        help='test with trained model')

    parser.add_argument('--epochs', type=int, default=12,
                        help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--loss', type=str, default='focal_loss', help='loss function')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataloaders
    print('data loading....')

    if args.dataset == 'LoveDa':
        train_set = LoveDa_loader.LoveDaSeg(root=args.data_path, split='Train', transform=True)
        val_set = LoveDa_loader.LoveDaSeg(root=args.data_path, split='Val', transform=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        print("LoveDa train data load success")

        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        print("LoveDa val data load success")

    elif args.dataset == 'GID-5':
        train_set = GID_loader.VOCClassSegBase(root=args.data_path, split='Train', transform=True)
        val_set = GID_loader.VOCClassSegBase(root=args.data_path, split='Val', transform=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        print("GID-5 train data load success")

        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        print("GID-5 val data load success")

    else:
        train_loader = val_loader = None
        NotImplementedError

    dataset_dir = {'LoveDa': ('Background', 'Building', 'Road',
                              'Water', 'Barren', 'Forest', 'Agricultural'),
                   'GID-5': ('other', 'built_up', 'farmland', 'forest', 'meadow', 'water')}

    num_classes = len(dataset_dir[args.dataset])

    dataloaders = (train_loader, val_loader)

    # network
    if args.model == 'fcn_resnet':
        resnet = fcn_resnet.RESNET().to(device)
        model = fcn_resnet.FCN8s(pretrained_net=resnet, n_class=num_classes, backbone='resnet50').to(device)

        print("The FCN ResNet model is defined")

    elif args.model == 'fcn_vgg':
        # vgg16 Model Calling
        vgg_model = models.VGGNet(requires_grad=True, pretrained=True).to(device)
        # FCN8s Model Calling
        model = models.FCN8s(pretrained_net=vgg_model, n_class=num_classes).to(device)
        print("The model is defined")

    elif args.model == 'FCNformer':
        backbone = swin.swin_t()
        model = FCNformer.FCN8s(pretrained_net=backbone, n_class=num_classes, backbone='swin-T')

    else:
        model = None
        NotImplementedError
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # resume the trained model
    if args.resume:
        model, optimizer = resume(args, model, optimizer)

    if args.test == 1:  # test mode, resume the trained model and test
        evaluate(args, model, val_loader, num_classes)
    else:  # train mode, train the network from scratch
        train(args, model, optimizer, dataloaders, num_classes)
        print('training finished')
