import os
import time

import torch
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import MIoU
from loss import loss
# from main import parse_args

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# arg = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_dir = {'cross entropy': torch.nn.CrossEntropyLoss(),
            'Focal loss': loss.FocalLoss()}

criterion = loss_dir['Focal loss']

train_loss = []
val_loss = []
train_acc = []
val_acc = []
val_miou = []

CLASSES = ('Background', 'Building', 'Road',
           'Water', 'Barren', 'Forest', 'Agricultural')


def train(args, model, optimizer, dataloaders, num_classes):
    train_loader, val_loader = dataloaders

    # total_count = torch.tensor([0.0]).to(device)
    # correct_count = torch.tensor([0.0]).to(device)
    total_loss = torch.tensor([0.0]).to(device)
    # miou_count = torch.tensor([0.0]).to(device)

    best_miou = 0.0

    print('Network training starts with {} epochs'.format(args.epochs))
    for epoch in range(args.epochs):
        model.train()

        iter_time = time.time()

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

        acc, miou = evaluate(args, model, val_loader, num_classes)

        print('\nAcc = %.2f' % acc, '% ', 'MIoU = %.2f' % miou, '%')

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


def evaluate(args, model, val_loader, num_classes):
    total_loss = torch.tensor([0.0]).to(device)
    total_count = torch.tensor([0.0]).to(device)
    correct_count = torch.tensor([0.0]).to(device)

    model.eval()

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

    total_loss /= len(val_loader)

    b, _, h, w = out.size()
    pred = out.permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, h, w)

    out_np = pred.cpu().detach().numpy().copy()

    labels_np = labels_tensor.cpu().detach().numpy().copy()

    acc, acc_cls, mean_iu, iu = MIoU.label_accuracy_score(labels_np, out_np, num_classes)

    acc = acc * 100
    mean_iu = mean_iu * 100

    val_loss.append(total_loss)
    val_acc.append(acc)
    val_miou.append(mean_iu)

    return acc, mean_iu


def resume(args, model, optimizer):
    checkpoint_path = './{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(
        checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    # load the model and the optimizer --------------------------------
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # -----------------------------------------------------------------
    print('Resume completed for the model\n')

    return model, optimizer
