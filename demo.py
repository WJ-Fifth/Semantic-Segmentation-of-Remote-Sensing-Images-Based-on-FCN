import predict
import matplotlib.pyplot as plt
from dataloaders import GID_loader
import torch
from Test_run import Test
import os

data_path = os.path.expanduser('./data/')


def platform(title, line_num=2, epoch_num=100, label_name=False, file_names=None):
    files = []
    labels = []
    x = []
    y = []

    for n in range(epoch_num):
        x.append(n + 1)

    for text in range(line_num):
        files.append("./measure/" + file_names[text] + ".txt")
        labels.append(file_names[text])
        file = open(files[text])
        y.clear()
        for line in file:
            line = line.strip('\n')
            if line == '':
                break
            y.append(float(line))
        file.close()
        if title == "MIoU":
            if text == 0:
                plt.plot(x, y, color='red', marker='*', ms=5, linestyle='-', label=labels[text])
            elif text == 1:
                plt.plot(x, y, color='blue', marker='o', ms=3, linestyle='-', label=labels[text])
            else:
                return "Invalid file"
        else:
            if text == 0:
                plt.plot(x, y, color='black', marker='x', ms=5, linestyle='-', label=labels[text])
            elif text == 1:
                plt.plot(x, y, color='red', linestyle='-.', label=labels[text])
            else:
                return "Invalid file"

    plt.legend()  # Making the legend work
    plt.title(title)  # Set title and font

    ax = plt.gca()  # Get the current axis position and move it
    ax.spines['right'].set_color('none')  # The right border property is set to none and is not displayed
    ax.spines['top'].set_color('none')  # The top border property is set to none and is not displayed
    plt.grid(True)  # Increase grid points
    if label_name:
        plt.ylabel(input("Please enter the Y-axis name: "))
    elif not label_name:
        plt.ylabel('Value')
    plt.xlabel('epoch')

    plt.show()


def main():
    batch_size = 32
    test_data = voc_loader.VOC2012ClassSeg(root=data_path, split='test', transform=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print("test data load success")

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    print("testing start")
    Test(0, test_loader)

    predict.main()
    platform(title="Loss", file_names=["train", "test"])
    platform(title="MIoU", file_names=["MIoU", "UNet_MIoU"])


if __name__ == "__main__":
    class_names = ['other', 'built_up', 'farmland', 'forest', 'meadow', 'water']

    # mean_iu = [69.42,
    #      85.41,
    #      78.54,
    #      70.25,
    #      6.57,
    #      6.10]
    # for i in range(6):
    #     print('class name: %s' % class_names[i], 'MIoU = %.2f' % mean_iu[i])
    # print('Total MIoU = : %.2f' % np.mean(mean_iu))
    main()
