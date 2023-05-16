# 用于合成数据输入的realsense采集到的，shape 800,848   手的数据是txt，背景是255
import os
import time
import shutil
from data_augmentation import *

background_path = r'D:\script\gesture\background_1'  # 背景文件夹
hand_path = r'D:\matting_data\synthesis_2\jin_classify1'  # crop出来的txt文件夹
syn_path = r'D:\script\gesture\chartlet\3_syn'  # 用来放置合成的图片

pic_num = 3000  # 生成多少张图片
length_interval = 5  # 手的框和手的最大最小值像差多少
final_size = [800, 848]  # 合成以后的图片大小

if_hand_resize = True
hand_length_range = [100, 250]  # 手的有效数值（非255）最长边的范围

if_background_resize = True
background_height_range = [0.5, 1]
background_width_range = [0.5, 1]

if_hand_rotate = True
hand_rotate_range = 180
if_background_rotate = True
background_rotate_range = 180

if_hand_mean_std = True
hand_mean_range = [0.7, 1.3]
hand_std_range = [0.7, 1.3]

if_background_mean_std = True
background_mean_range = [0.7, 1.3]
background_std_range = [0.7, 1.3]

if os.path.exists(syn_path):
    shutil.rmtree(syn_path)
os.mkdir(syn_path)

background_list = os.listdir(background_path)
hand_list = os.listdir(hand_path)

for i in range(pic_num):
    background_index = np.random.randint(0, len(background_list))
    hand_index = np.random.randint(0, len(hand_list))
    # 读取数据
    background = cv2.imread(background_path + '/' + background_list[background_index], 0)
    hand = np.loadtxt(hand_path + '/' + hand_list[hand_index], dtype=np.uint8,delimiter=' ')

    # 手的旋转
    if if_hand_rotate:
        hand = rotate_total(hand, hand_rotate_range).astype(np.uint8)

    # 背景旋转
    if if_background_rotate:
        background = rotate_part(background, background_rotate_range)

    # 手的有效范围的缩放
    if if_hand_resize:
        hand = hand_resize(hand, hand_length_range)

    # 背景的缩放
    if if_background_resize:
        background = background_resize(background, background_height_range, background_width_range, final_size)

    # 手的均值和方差
    if if_hand_mean_std:
        hand = hand_mean_std(hand, hand_mean_range, hand_std_range)

    # 背景的均值和方差
    if if_background_mean_std:
        background = background_mean_std(background, background_mean_range, background_std_range)

    # 贴图
    background_h, background_w = background.shape
    hand_h, hand_w = hand.shape
    new_w = np.random.randint(0, background_w - hand_w)
    new_h = np.random.randint(0, background_h - hand_h)

    # 获得手和手势框的有效范围
    hand_h_min, hand_w_min, hand_h_max, hand_w_max = 100000, 100000, 0, 0
    for h in range(hand_h):
        for w in range(hand_w):
            if hand[h, w] != 255:
                hand_h_min = min(hand_h_min, h)
                hand_w_min = min(hand_w_min, w)
                hand_h_max = max(hand_h_max, h)
                hand_w_max = max(hand_w_max, w)
    hand_h_min = hand_h_min + new_h - length_interval
    hand_h_max = hand_h_max + new_h + length_interval
    hand_w_min = hand_w_min + new_w - length_interval
    hand_w_max = hand_w_max + new_w + length_interval
    hand_h_min = max(0, hand_h_min)
    hand_h_max = min(hand_h_max, background_h)
    hand_w_min = max(0, hand_w_min)
    hand_w_max = min(hand_w_max, background_w)

    img = background.copy()
    # 覆盖
    for h in range(hand_h):
        for w in range(hand_w):
            if hand[h, w] != 255:
                img[new_h + h, new_w + w] = hand[h, w]

    img = cv2.blur(img, (3, 3))
    cv2.rectangle(img, (hand_w_min, hand_h_min), (hand_w_max, hand_h_max), 255, 2)
    cv2.namedWindow = cv2.WINDOW_NORMAL
    w, h = img.shape
    img = cv2.resize(img, (int(w / 1.5), int(h / 1.5)))
    cv2.imwrite(filename="%s\\%d.jpg" % (syn_path, i), img=img)
    # cv2.imshow('final', img)
    #
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #
    #     break
cv2.destroyAllWindows()
