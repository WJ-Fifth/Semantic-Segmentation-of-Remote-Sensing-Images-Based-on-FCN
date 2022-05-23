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
            labels.append(input("曲线名："))
        elif not label_name:
            labels.append(file_name+"曲线")
        file = open(files[text])
        y.clear()
        for line in file:
            line = line.strip('\n')
            if line == '':
                break
            y.append(float(line))
        file.close()
        if text == 2:
            plt.plot(x, y, color='red', marker='*', ms=5, linestyle='-', label=labels[text])
        elif text == 3:
            plt.plot(x, y, color='blue', marker='o', ms=3, linestyle='-', label=labels[text])
        elif text == 0:
            plt.plot(x, y, color='black', marker='x', ms=5, linestyle='-', label=labels[text])
        elif text == 1:
            plt.plot(x, y, color='red', linestyle='-.', label=labels[text])
        else:
            return "Invalid file"

    plt.legend()  # 让图例生效
    plt.title(title)  # 设置标题及字体
    # plt.legend(handles=[line_1, line_2], labels=['Acc曲线图', 'MIoU曲线图'])
    # plt.plot(acc_x, acc_y, 'ro') #离散的点
    ax = plt.gca()  # 获取当前坐标轴位置并移动
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.grid(True)  # 增加格点
    if label_name:
        plt.ylabel(input("请输入Y轴名称："))
    elif not label_name:
        plt.ylabel('Value-数值')
    plt.xlabel('epoch')

    plt.show()

def Random():
    n = []
    count = 0.231
    for i in range(85):
        count = count + random.uniform(-0.003, 0.0015)
        print("%.3f" % count)


if __name__ == '__main__':
    # title = input("请输入图标名：")
    line_num = int(input("请输入文件数目："))
    epoch_num = int(input("请输入迭代次数："))
    platform(title=None, line_num=line_num, epoch_num=epoch_num, label_name=True)

