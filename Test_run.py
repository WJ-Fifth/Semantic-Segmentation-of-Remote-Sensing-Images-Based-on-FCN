import os
import torch
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import models
import voc_loader
import loss
from torch.optim import Adam
import torch.nn as nn
import MIoU

CUDA_LAUNCH_BLOCKING = 1

batch_size = 4
learning_rate = 0.001
epoch_num = 30
n_class = 6
all_train_loss = []
train_file = "measure/train_loss.txt"

all_test_loss = []
test_file = "measure/test_loss.txt"

test_Acc = []
Acc_file = "measure/acc.txt"

test_mIou = []
MIoU_file = "measure/MIoU.txt"

test_Iou = []
best_test_loss = np.inf
pretrained = 'reload'
use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = os.path.expanduser('E:/FCN_Code/data/')
# vgg16 Model Calling
vgg_model = models.VGGNet(requires_grad=True, pretrained=False)
# FCN8s Model Calling
fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

fcn_model.load_state_dict(torch.load('E:/FCN_Code/model/model30.pth'))  # 上次训练的参数

if use_cuda:
    fcn_model.cuda()

# criterion = nn.CrossEntropyLoss()
criterion = loss.FocalLoss()

optimizer = Adam(fcn_model.parameters())


def Test(epoch):
    fcn_model.eval()
    total_loss = 0.0
    for batch_idx, (imgs, labels) in enumerate(test_loader):
        N = imgs.size(0)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        imgs = Variable(imgs)  # , volatile=True
        labels_out = Variable(labels)  # , volatile=True
        optimizer.zero_grad()  # Clear the residual update parameter values from the previous step
        with torch.no_grad():
            out = fcn_model(imgs)
        # out = torch.nn.functional.softmax(out, dim=1)
        loss = criterion(out, labels_out)
        loss /= N
        all_test_loss.append(loss)
        total_loss += loss.item()

        rate = (batch_idx + 1) / len(test_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtest loss:  {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        # if (batch_idx + 1) % 3 == 0:
        #     print('\rtest epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch,
        #                                                                epoch_num, batch_idx, len(test_loader),
        #                                                                total_loss / (batch_idx + 1)))

        # Confirm that the obtained loss is a valid value
        assert total_loss is not np.nan  # Determine that the loss ratio is not null
        assert total_loss is not np.inf  # Determine that the loss rate is not infinite

    total_loss /= len(test_loader)

    with open(test_file, "a+") as file:
        file.write('%.3f' % total_loss + '\n')
        file.close()

    print()
    print('test epoch  [%d/%d] average_loss %.5f' % (epoch + 1, epoch_num, total_loss))
    # print('test epoch [%d/%d]: ' % (epoch + 1, epoch_num))
    global best_test_loss
    if best_test_loss > total_loss:
        best_test_loss = total_loss
        print('the best loss!')
        # fcn_model.save('SBD.pth')

    b, _, h, w = out.size()
    pred = out.permute(0, 2, 3, 1).contiguous().view(-1, n_class).max(1)[1].view(b, h, w)
    # print("pred:", pred.type(), pred.shape)
    # print("out:", out.type(), out.shape)
    # print("label:", labels_out.type(), labels_out.shape)
    out_np = pred.cpu().detach().numpy().copy()
    # out_np = out_np.argmin(out_np, axis=1)
    # print(out_np.shape)
    labels_np = labels_out.cpu().detach().numpy().copy()
    # labels_np = np.argmin(labels_np, axis=1)
    # print(labels_np.shape)
    acc, acc_cls, mean_iu, iu = MIoU.label_accuracy_score(labels_np, out_np, n_class)

    test_Acc.append(acc)
    test_mIou.append(mean_iu)
    # test_Iou.append(iu)

    acc = acc * 100
    mean_iu = mean_iu * 100
    print('Acc = %.2f' % acc, '% ', 'MIoU = %.2f' % mean_iu, '%')
    with open(Acc_file, "a+") as file:
        file.write('%.2f' % acc + '\n')
        file.close()

    with open(MIoU_file, "a+") as file:
        file.write('%.2f' % mean_iu + '\n')
        file.close()


if __name__ == '__main__':
    print('data loading....')

    test_data = voc_loader.VOC2012ClassSeg(root=data_path, split='test', transform=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print("test data load success")

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    print("training start")
    Test(30)
