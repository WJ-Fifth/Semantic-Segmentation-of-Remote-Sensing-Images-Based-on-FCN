import os
import time

import torch
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
from loss import focal, dice
from Toolkit import metric

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select the loss function required for training
loss_dir = {'cross entropy': torch.nn.CrossEntropyLoss(),
            'Focal loss': focal.FocalLoss(),
            'Dice loss': dice.DiceLoss()}
criterion = loss_dir['Dice loss']

train_loss = []
val_loss = []
train_acc = []
val_acc = []
val_miou = []

CLASSES = ('Background', 'Building', 'Road',
           'Water', 'Barren', 'Forest', 'Agricultural')


def train(args, model, optimizer, dataloaders, num_classes):
    """
    :param args: configs
    :param model: Selected semantic segmentation models
    :param optimizer: Selected Optimizer, AdamW
    :param dataloaders: Completed loaded datasets GID-5 or LoveDA
    :param num_classes: 6 in GID-5 & 7 in LoveDA
    :return: 'None' just save the checkpoint and Training process.
    """
    train_loader, val_loader = dataloaders

    total_loss = torch.tensor([0.0]).to(device)

    best_miou = 0.0

    print('Network training starts with {} epochs'.format(args.epochs))
    for epoch in range(args.epochs):
        model.train()

        iter_time = time.time()
        print('----------------------------------------------------------')
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs_tensor = Variable(imgs.to(device))
            labels_tensor = Variable(labels.to(device))
            out = model(imgs_tensor)

            loss = criterion(out, labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            rate = (batch_idx + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

            # Confirm that the obtained loss is a valid value
            assert total_loss is not np.nan  # Determine that the loss ratio is not null
            assert total_loss is not np.inf  # Determine that the loss rate is not infinite

        total_loss /= len(train_loader)

        print()
        print('train epoch [%d/%d] average_loss %.5f|time:{%.2f}'
              % (epoch + 1, args.epochs, total_loss, time.time() - iter_time))
        train_loss.append(total_loss.item())
        acc, miou = evaluate(args, model, val_loader, num_classes)

        print('\nAcc = %.2f' % acc * 100, '% ', 'MIoU = %.2f' % miou * 100, '%')

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, './checkpoints/{}_checkpoint.pth'.format(args.exp_id))
            print("The best model is saved!")

    np.save("./train_loss.npy", train_loss)
    np.save("./val_loss.npy", val_loss)
    np.save("./val_acc.npy", val_acc)
    np.save("./val_miou.npy", val_miou)
    torch.save(
        {
            'epoch': 30,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, './checkpoints/{}_final_checkpoint.pth'.format(args.exp_id))


def evaluate(args, model, val_loader, num_classes):
    total_loss = torch.tensor([0.0]).to(device)
    # total_count = torch.tensor([0.0]).to(device)
    # correct_count = torch.tensor([0.0]).to(device)

    model.eval()

    #
    SegmentationMetric = metric.SegmentationMetric(num_classes)
    SegmentationMetric.reset()

    acc = 0.0
    mean_iu = 0.0

    class_iou = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for batch_idx, (imgs, labels) in enumerate(val_loader):
        imgs_tensor = Variable(imgs.to(device))
        labels_tensor = Variable(labels.to(device))

        with torch.no_grad():
            out = model(imgs_tensor)

        loss = criterion(out, labels_tensor)
        total_loss += loss.item()

        rate = (batch_idx + 1) / len(val_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rval loss:  {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        # Confirm that the obtained loss is a valid value
        assert total_loss is not np.nan  # Determine that the loss ratio is not null
        assert total_loss is not np.inf  # Determine that the loss rate is not infinite

        SegmentationMetric.update(preds=out, labels=labels_tensor)
        pixAcc, mIoU, IoU = SegmentationMetric.get()

        acc += pixAcc
        mean_iu += mIoU
        class_iou = [i + j for i, j in zip(class_iou, IoU)]
        # print(class_iou)

    total_loss /= len(val_loader)

    b, _, h, w = out.size()
    # pred = out.permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, h, w)

    acc = acc / len(val_loader)
    mean_iu = mean_iu / len(val_loader)
    class_iou = [x / len(val_loader) for x in class_iou]

    val_loss.append(total_loss.item())
    val_acc.append(acc)
    val_miou.append(mean_iu)

    print()
    print("MIoU=", mean_iu)
    print("ACC=", acc)
    print(class_iou)
    return acc, mean_iu


def resume(args, model, optimizer):
    checkpoint_path = './checkpoints/{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(
        checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    # load the model and the optimizer --------------------------------
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # -----------------------------------------------------------------
    print('Resume completed for the model\n')

    return model, optimizer
