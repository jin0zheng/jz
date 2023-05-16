#!/usr/bin/env python3
import os

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
# import rosbag
import pyrosbag as rosbag
from scipy.signal import savgol_filter

# 常量设置
LINE_WIDTH = 0.4  # 绘图线宽
SKIP_ITEM_NUM = 1  # 间隔固定条数数据
SEQ_OFFSET = 0  # 从第一条之后的某一条数据开始
WINDOW_SIZE = 60  # 数据平滑窗口的数据数量

txt_flag = False  # 是否为txt文件
bag_flag = False  # 是否为bag文件
imu_data = None
item_count = 0
interval_count = 0


def limit_filter(data, top=4):
    '''
        限幅滤波法（又称程序判断滤波法）  
        A、方法：  根据经验判断，确定两次采样允许的最大偏差值（设为top）每次检测到新值时判断：如果本次值与上次值之差<=top,则本次值有效  如果本次值与上次值之差>top,则本次值无效,放弃本次值,用上次值代替本次值  
        B、优点：  能有效克服因偶然因素引起的脉冲干扰  
        C、缺点：  无法抑制那种周期性的干扰  平滑度差
    '''
    flag_arr = np.argwhere(data < 0)
    temp_data = abs(data)
    for i in range(1, len(temp_data)):
        if abs(temp_data[i] - temp_data[i - 1]) > top:
            temp_data[i] = temp_data[i - 1]
    temp_data[flag_arr] *= -1
    return temp_data


def reform_time(data):
    # reform time
    start_sec = data[0][0]
    start_nsec = data[0][1]
    t = np.zeros(data.shape[0], dtype='float32')
    if txt_flag:
        for i in range(data.shape[0]):
            t[i] = (data[i, 0] - start_sec) + (data[i, 1] - start_nsec) / 1000000.0
    elif bag_flag:
        for i in range(data.shape[0]):
            t[i] = (data[i, 0] - start_sec) + (data[i, 1] - start_nsec) / 1000000000.0
    return t


def load_file():
    # get path
    global path
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        if 'data' in dirs:  # 只查找data文件夹
            continue
        for file in files:
            if '.txt' in file:  # 判断是否为txt文件
                global txt_flag
                txt_flag = True
            elif '.bag' in file:  # 判断是否为bag文件
                global bag_flag
                bag_flag = True
            else:
                print("该文件类型无法识别！")
            global fig_name
            fig_name = file.split(".")[0]  # 获取.bag文件名
            file_path = path + "/data"
            filename = os.path.normpath(os.path.join(file_path, file))
            print(filename)
    return filename


def readBag(file):
    global interval_count
    global item_count
    data = None
    # read from bag file
    bag = rosbag.Bag(file, "r")
    info = bag.get_type_and_topic_info()
    """
    info数据格式
    TypesAndTopicsTuple(msg_types={'sensor_msgs/Imu': '6a62c6daae103f4ff57a132d6f95cec2'}, \
                        topics={'/imu': TopicTuple(msg_type='sensor_msgs/Imu', message_count=1670587,\
                                connections=1, frequency=3792.3182640144664)}) 
    """
    message_count = info.topics["/imu"].message_count
    print('\n数据条数为:', message_count)
    # print(info) #打印bag文件信息

    # 读取并保存数据
    is_first_item = True
    for topic, msg, t in bag.read_messages(['/imu']):  # topic是信息的话题；msg是具体的消息数据；t表示时间戳
        if (is_first_item):  # 只执行一次，获得第一条数据的序列号
            first_item_seq = msg.header.seq
            next_item_seq = first_item_seq + SEQ_OFFSET
            last_item_seq = first_item_seq + message_count - 1
            message_count = message_count - SEQ_OFFSET
            print('\n第一条数据的序列号为：', first_item_seq)
            is_first_item = False

        cur_item_seq = msg.header.seq
        if interval_count >= SKIP_ITEM_NUM:  # 此时应该采用的数据可能缺失，移动到下一条数据
            next_item_seq = cur_item_seq
            print('此处有数据缺失：', interval_count)
            interval_count = 0  # 该条数据采用后，变量归零
        interval_count += 1  # 上一个采用的数据后已遍历的数据条数

        if cur_item_seq + SKIP_ITEM_NUM > last_item_seq:
            break

        if ((cur_item_seq == next_item_seq) & (next_item_seq < last_item_seq)):  # 读取/imu主题
            print(cur_item_seq, next_item_seq)
            next_item_seq += SKIP_ITEM_NUM
            item = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            item[0, 0] = t.secs
            item[0, 1] = t.nsecs
            item[0, 2] = msg.linear_acceleration.x
            item[0, 3] = msg.linear_acceleration.y
            item[0, 4] = msg.linear_acceleration.z
            item[0, 5] = msg.angular_velocity.x
            item[0, 6] = msg.angular_velocity.y
            item[0, 7] = msg.angular_velocity.z
            item[0, 8] = msg.linear_acceleration_covariance[0]  # IMU内置的温度数据

            item_count += SKIP_ITEM_NUM
            print("当前条目：%s / %s" % (item_count, message_count))
            sg.one_line_progress_meter('IMU数据可视化', item_count, message_count, '原始数据处理进度')
            if data is None:
                data = item
            else:
                data = np.append(data, item, axis=0)
    return data


def readTxt(filename):
    data = None
    item_count = 0
    file = open(filename, 'r')
    column_name = file.readline()
    file_data = file.readlines()
    for row in file_data:
        line_tmp = row.split(' ')
        str_tmp = line_tmp[0].strip('\n')  # 去掉换行符，成为字符串
        column = str_tmp.split('\t')
        list_tmp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # data = list_tmp
        i = 1
        for item in column:
            if item:
                list_tmp[0][i] = item
                i += 1
        item_count += 1
        if data is None:
            data = list_tmp
        else:
            data = np.append(data, list_tmp, axis=0)
        # print("data", data)
        # print('\n')
    return data


# 绘制数据子图的函数(有平滑，温度)
def subplotIMU(data, data_smooth, fig, fig_num, sensor_type, axis, data_col, temp=True):
    label1 = sensor_type + " " + axis
    label2 = "smoothed " + sensor_type + " " + axis
    if g_axis == data_col or (g_axis == data_col - 3):
        label1 += " (g)"
        label2 += " (g)"

    if sensor_type == "accl":
        title = axis + " linear acceleration"
        ylabel = "linear acceleration [m/s^2]"
    elif sensor_type == "gyro":
        title = axis + " angular velocity"
        ylabel = "angular velocity [s^-1]"
    elif sensor_type == "mag":
        title = axis + " mag"

    ax1 = fig.add_subplot(fig_num)
    ln1 = ax1.plot(t, data[:, data_col], 'dodgerblue', linewidth=LINE_WIDTH, label=label1)
    ln2 = ax1.plot(t, data_smooth[:, data_col], 'cyan', linewidth=LINE_WIDTH, label=label2)
    ax1.set_ylabel(ylabel)
    ax2 = ax1.twinx()
    ln3 = ax2.plot(t, data[:, len(data[0]) - 1], 'red', linewidth=LINE_WIDTH, label="temp")
    ax2.set_ylabel("Temperature [℃]")
    plt.title(title)
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.6)
    plt.xlabel("time [s]")
    # 合并图例
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, frameon=False)


# 绘制数据到一个图的函数(有平滑，温度)
def plotIMU(data, data_smooth, fig, sensor_type, data_col, temp=True):
    data_x_col = data_col
    data_y_col = data_col + 1
    data_z_col = data_col + 2
    label1 = sensor_type + " x"
    label2 = sensor_type + " y"
    label3 = sensor_type + " z"
    if sensor_type == "accl":
        title = " linear acceleration"
        ylabel = "linear acceleration [m/s^2]"
    elif sensor_type == "gyro":
        title = " angular velocity"
        ylabel = "angular velocity [s^-1]"
    elif sensor_type == "mag":
        title = " mag"

    if g_axis == data_col or (g_axis == data_col - 3):
        label1 += " (g)"
        data_x_col = len(data_smooth)
    elif g_axis == data_col + 1 or g_axis == data_col - 2:
        label2 += " (g)"
        data_y_col = len(data_smooth)
    elif g_axis == data_col + 2 or g_axis == data_col - 1:
        label3 += " (g)"
        data_z_col = len(data_smooth)

    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(t, data_smooth[:, data_x_col], 'b', linewidth=LINE_WIDTH, label=label1)
    ln2 = ax1.plot(t, data_smooth[:, data_y_col], 'cyan', linewidth=LINE_WIDTH, label=label2)
    ln3 = ax1.plot(t, data_smooth[:, data_z_col], 'g', linewidth=LINE_WIDTH, label=label3)
    ax1.set_ylabel(ylabel)
    ax2 = ax1.twinx()
    ln4 = ax2.plot(t, imu_data[:, 8], 'red', linewidth=LINE_WIDTH, label="temp")
    ax2.set_ylabel("Temperature [℃]")
    plt.title(title)
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.6)
    plt.xlabel("time [s]")
    # 合并图例
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0, frameon=False)


# 绘制数据子图的函数(无平滑，温度)+
def subplotIMU(data, data_smooth, fig, fig_num, sensor_type, axis, data_col):
    label1 = sensor_type + " " + axis
    label2 = "smoothed " + sensor_type + " " + axis
    # if g_axis == data_col or (g_axis == data_col-3):
    #     label1 += " (g)"
    #     label2 += " (g)"

    if sensor_type == "accl":
        title = axis + " linear acceleration"
        ylabel = "linear acceleration [m/s^2]"
    elif sensor_type == "gyro":
        title = axis + " angular velocity"
        ylabel = "angular velocity [s^-1]"
    elif sensor_type == "mag":
        title = axis + " mag"

    ax1 = fig.add_subplot(fig_num)
    ln1 = ax1.plot(t, data[:, data_col], 'dodgerblue', linewidth=LINE_WIDTH, label=label1)
    ln2 = ax1.plot(t, data_smooth[:, data_col], 'cyan', linewidth=LINE_WIDTH, label=label2)
    ax1.set_ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.6)
    plt.xlabel("time [s]")
    # 合并图例
    lns = ln1 + ln2  # + ln3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, frameon=False)


# 绘制数据到一个图的函数(无平滑，温度)
def plotIMU(data, data_smooth, fig, sensor_type, data_col):
    data_x_col = data_col
    data_y_col = data_col + 1
    data_z_col = data_col + 2
    label1 = sensor_type + " x"
    label2 = sensor_type + " y"
    label3 = sensor_type + " z"
    if sensor_type == "accl":
        title = " linear acceleration"
        ylabel = "linear acceleration [m/s^2]"
    elif sensor_type == "gyro":
        title = " angular velocity"
        ylabel = "angular velocity [s^-1]"
    elif sensor_type == "mag":
        title = " mag"


    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(t, data_smooth[:, data_x_col], 'b', linewidth=LINE_WIDTH, label=label1)
    ln2 = ax1.plot(t, data_smooth[:, data_y_col], 'cyan', linewidth=LINE_WIDTH, label=label2)
    ln3 = ax1.plot(t, data_smooth[:, data_z_col], 'g', linewidth=LINE_WIDTH, label=label3)
    ax1.set_ylabel(ylabel)
    # ax2 = ax1.twinx()
    # ln4 = ax2.plot(t, imu_data[:,len(data_smooth)], 'red', linewidth=LINE_WIDTH, label="temp")
    # ax2.set_ylabel("Temperature [℃]")
    plt.title(title)
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.6)
    plt.xlabel("time [s]")
    # 合并图例
    lns = ln1 + ln2 + ln3  # + ln4
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0, frameon=False)


# # 限幅滤波，滤除尖峰
# for i in range(2,5):
#     imu_data[:,i] = limit_filter(imu_data[:,i], top=0.3)
# for i in range(5,8):
#     imu_data[:,i] = limit_filter(imu_data[:,i], top=0.5e-2)

def find_g(data):
    global g_axis
    for i in range(2, 5):  # len(data[0]
        if (abs(data[:, i][0]) > 5):
            g_axis = i
    return g_axis


def smooth_data(data):
    # 数据平滑
    data_smooth = data.copy()
    for i in range(2, len(data[0]) - 1):
        if (i < 5) & (abs(data[:, i][0]) > 5):  # 找重力所在的坐标轴
            g_col = len(data[0]) - 1  # 平滑后的数组中存放减去重力加速度后的数据的列数，放在最右边一列
            g_norm = 9.7 * abs(data[:, i][0]) / data[:, i][0] # -9.7,获取标准的中立加速度
            data_smooth[:, i] = savgol_filter(data[:, i], WINDOW_SIZE + 1, 3)  # 窗口长度、拟合阶数
            data_smooth[:, g_col] = savgol_filter(data[:, i] - g_norm, WINDOW_SIZE + 1, 3)  # 窗口长度、拟合阶数
        data_smooth[:, i] = savgol_filter(data[:, i], WINDOW_SIZE + 1, 3)  # 窗口长度、拟合阶数
    return data_smooth


filename = load_file()
if txt_flag:
    imu_data = readTxt(filename)
elif bag_flag:
    imu_data = readBag(filename)
else:
    print("该文件类型无法识别！")

print("传感器数据：", imu_data)

t = reform_time(imu_data)
print("时间：", t)

g_axis = find_g(imu_data)
print(imu_data)
imu_data_smooth = smooth_data(imu_data)

############################### plot linear accelaration
fig1 = plt.figure(figsize=(12, 9), dpi=200)
fig1.tight_layout()
fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, \
                     wspace=None, hspace=0.5)  # 设置子图间隔
subplotIMU(imu_data, imu_data_smooth, fig1, 311, "accl", "x", 2)
subplotIMU(imu_data, imu_data_smooth, fig1, 312, "accl", "y", 3)
subplotIMU(imu_data, imu_data_smooth, fig1, 313, "accl", "z", 4)
plt.savefig(path + '/' + fig_name + '_linear-acceleration.png')

############################### plot angular velocity in one plot
fig1 = plt.figure(figsize=(12, 3), dpi=200)
fig1.tight_layout()
fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)  # 设置子图间隔
plotIMU(imu_data, imu_data_smooth, fig1, "accl", 2)
plt.savefig(path + '/' + fig_name + '_all-linear-acceleration.png')

############################### plot angular velocity
fig1 = plt.figure(figsize=(12, 9), dpi=200)
fig1.tight_layout()
fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, \
                     wspace=None, hspace=0.5)  # 设置子图间隔
subplotIMU(imu_data, imu_data_smooth, fig1, 311, "gyro", "x", 5)
subplotIMU(imu_data, imu_data_smooth, fig1, 312, "gyro", "y", 6)
subplotIMU(imu_data, imu_data_smooth, fig1, 313, "gyro", "z", 7)
plt.savefig(path + '/' + fig_name + '_angular-velocity.png')

############################### plot angular velocity in one plot
fig1 = plt.figure(figsize=(12, 3), dpi=200)
fig1.tight_layout()
fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, \
                     wspace=None, hspace=0.5)  # 设置子图间隔
plotIMU(imu_data, imu_data_smooth, fig1, "gyro", 5)
plt.savefig(path + '/' + fig_name + '_all-angular-velocity.png')

if bag_flag:
    bag.close()
# /bin/python3 /home/pyj/文档/visualizeRawBagData/visualizeRawBagData.py
