import os
import torch
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
from model import models
from dataset import voc_loader
from loss import loss
from torch.optim import Adam

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
data_path = os.path.expanduser('data/')
# vgg16模型调用
vgg_model = models.VGGNet(requires_grad=True)
# FCN8s模型调用
fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

fcn_model.load_state_dict(torch.load('model/model20.pth'))  # 上次训练的参数

# FCN32s模型调用
# fcn_model = models.FCN32s(pretrained_net=vgg_model, n_class=n_class)
if use_cuda:
    fcn_model.cuda()

# criterion = nn.CrossEntropyLoss()
criterion = loss.FocalLoss()

optimizer = Adam(fcn_model.parameters())


def train(epoch):
    fcn_model.train()  # train mode

    total_loss = 0.0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        N = imgs.size(0)  # 2
        # print(N)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()

        imgs_tensor = Variable(imgs)  # torch.Size([2, 3, 640, 640])  设置为变量，可随反向传播更新参数
        labels_tensor = Variable(labels)  # torch.Size([2, 640, 640])
        out = fcn_model(imgs_tensor)  # torch.Size([2, 6, 640, 640])
        # out = torch.nn.functional.softmax(out, dim=1)
        loss = criterion(out, labels_tensor)
        loss /= N  # 四张图片的
        all_train_loss.append(loss)
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # update all arguments  将参数更新值施加到 net 的 parameters 上
        total_loss += loss.item()  # return float

        rate = (batch_idx + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        # if batch_idx % 20 == 0:
        #     print('train epoch [%d/%d], iter[%d/%d], lr %.4f, aver_loss %.5f' % (epoch, epoch_num,
        #                                                                          batch_idx, len(train_loader),
        #                                                                          learning_rate,
        #                                                                          total_loss / (batch_idx + 1)))
        # 确认得到的loss是有效值
        assert total_loss is not np.nan  # 判断损失率不为空
        assert total_loss is not np.inf  # 判断损失率不为无穷大

    total_loss /= len(train_loader)
    print()
    print('train epoch [%d/%d] average_loss %.5f' % (epoch + 1, epoch_num, total_loss))

    # model save

    if (epoch + 1) % 5 == 0:
        torch.save(fcn_model.state_dict(), './model/model%d.pth' % (epoch + 1))
        print("save the model")

    if epoch == 0:
        torch.save(fcn_model.state_dict(), './model/model%d.pth' % epoch)
        print("save the model")

    #  保存损失函数
    with open(train_file, "a+") as file:
        file.write('%.3f' % total_loss + '\n')
        file.close()
    all_train_loss.append(total_loss)


if __name__ == '__main__':
    print('data loading....')
    train_data = voc_loader.VOC2012ClassSeg(root=data_path, split='train', transform=True)
    # print(train_data.files)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    print("train data load success")

    test_data = voc_loader.VOC2012ClassSeg(root=data_path, split='test', transform=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print("test data load success")

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    print("training start")
    for epoch in range(25, epoch_num):
        train(epoch)
        # test(epoch)
