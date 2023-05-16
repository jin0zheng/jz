# -*- coding: utf-8 -*-
import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

path = 'left-swipe_positive_0.txt'
op = open(path)
labels = op.readline()
datas = [i for i in op.readlines() if 'imu' not in i]
datas = np.loadtxt(datas)
dataframe = pd.DataFrame(datas, columns=labels.split())
dataframe.PhoneTime = dataframe.PhoneTime - dataframe.PhoneTime[0]
gyro = 0
if gyro:
    dataframe['GYRO'] = (dataframe.ACC_X.apply(np.square) + dataframe.ACC_Y.apply(np.square) + dataframe.ACC_Z.apply(
        np.square)).apply(np.sqrt)
    temp = dataframe.pivot_table(index='PhoneTime', values='GYRO')
    temp.plot()
    plt.show()
smooth = 0
if smooth:
    dataframe_new = dataframe.copy()
    for i in dataframe_new.columns[2:8]:
        WINDOW_SIZE = 100  # 数据平滑窗口的数据数量
        dataframe_new[i] = savgol_filter(dataframe_new[i], WINDOW_SIZE + 1, 3)  # 窗口长度、拟合阶数
for k in [['ACC_X', 'ACC_Y', 'ACC_Z'], ['GYRO_X', 'GYRO_Y', 'GYRO_Z']]:
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    n = 0
    for i in dataframe[k]:
        temp = pd.pivot_table(dataframe, index='PhoneTime', values=i)
        temp.plot(ax=axs[n], title=i)
        n += 1
    plt.xticks(rotation=0)
    plt.subplots_adjust(hspace=1, wspace=0.4)
    file_name = os.path.basename(path).split('.')[0] + '_' + k[0].split('_')[0] + '.png'
    plt.savefig(file_name)
