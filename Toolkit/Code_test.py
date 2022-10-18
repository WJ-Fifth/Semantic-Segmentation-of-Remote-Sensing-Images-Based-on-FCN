import matplotlib.pyplot as plt
from pylab import *
import random

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文

files = []
labels = []
x = []
y = []


def platform(title, line_num, epoch_num, label_name=False):
    for n in range(epoch_num):
        x.append(n + 1)
    for text in range(line_num):
        file_name = input("please input the file name:")
        files.append("./measure/" + file_name + ".txt")
        if label_name:
            labels.append(input("line name："))
        elif not label_name:
            labels.append(file_name)
        file = open(files[text])
        y.clear()
        for line in file:
            line = line.strip('\n')
            if line == '':
                break
            y.append(float(line))
        file.close()
        if text == 0:
            plt.plot(x, y, color='red', marker='*', ms=5, linestyle='-', label=labels[text])
        elif text == 1:
            plt.plot(x, y, color='blue', marker='o', ms=3, linestyle='-', label=labels[text])
        # elif text == 0:
        #     plt.plot(x, y, color='black', marker='x', ms=5, linestyle='-', label=labels[text])
        # elif text == 1:
        #     plt.plot(x, y, color='red', linestyle='-.', label=labels[text])
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


if __name__ == '__main__':
    line_num = int(input("Please enter the number of files:"))
    epoch_num = int(input("Please enter the number of iterations:"))
    platform(title=None, line_num=line_num, epoch_num=epoch_num, label_name=True)
