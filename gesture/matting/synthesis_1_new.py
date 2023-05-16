# 根据输入的黑幕图片和标注，将手的方框抠出来并且进行长宽不变的缩放，长边值为缩放
import os
import cv2
import numpy as np
import shutil

# 输入就是自己采集到的的数据，每个文件夹分为anno和img两个文件夹
input_name = 'shiyan'
input_path = 'D:/matting_data/' + input_name
# 输出
out_img_path = 'D:/matting_data/synthesis_1_img'  # 手部的img数据
out_txt_path = 'D:/matting_data/synthesis_1_txt'  # 手部的txt数据

suofang = 180  # 首先缩放到的大小,这里要取偶数，因为在自适应阈值的时候,使用的blockSize为suofang+1,要是一个奇数
c_num = 0  # 这个是唯一需要调的参数，取的阈值数值为mean-c_num，也就是说c_num越大，阈值越低，手的范围就越大
blocksize = 181  # 滑窗的大小
if_show = 0  # 是不是看一眼
if_blank_ok = 0  # 允许手内部存在小空白
gesture_num_list_blank_ok = [20, 27]  # 哪几种手里里面允许存在小空白

if not os.path.exists(out_txt_path):
    os.mkdir(out_txt_path)
if not os.path.exists(out_img_path):
    os.mkdir(out_img_path)

if not os.path.exists(out_txt_path + '/' + input_name):
    os.mkdir(out_txt_path + '/' + input_name)
if not os.path.exists(out_img_path + '/' + input_name):
    os.mkdir(out_img_path + '/' + input_name)

img_list = os.listdir(input_path + '/img')  # 所有需要挑选的图片
folder_list = os.listdir(out_img_path + '/' + input_name)
img_already_done = []  # 已经挑选好的图片
for folder_name in folder_list:
    img_already_done += os.listdir(out_img_path + '/' + input_name + '/' + folder_name)

if os.path.exists(out_img_path + '/' + input_name + '/' + str(c_num)):
    print(str(c_num) + ' already done')
else:
    os.mkdir(out_img_path + '/' + input_name + '/' + str(c_num))
    if not os.path.exists(out_txt_path + '/' + input_name + '/' + str(c_num)):
        os.mkdir(out_txt_path + '/' + input_name + '/' + str(c_num))

    for img_name in img_list:
        if img_name in img_already_done:
            continue
        img = cv2.imread(input_path + '/img/' + img_name, 0)
        hand_bbox = np.loadtxt(input_path + '/anno/' + img_name[:-3] + 'txt').reshape(-1, 4).astype(int)
        x1, y1, x2, y2 = hand_bbox[0]

        hand_img = img[y1:(y2 + 1), x1:(x2 + 1)]

        h, w = hand_img.shape
        if h >= w:
            new_h = suofang
            new_w = int(suofang / h * w)
        else:
            new_w = suofang
            new_h = int(suofang / w * h)
        hand_img = cv2.resize(hand_img, (new_w, new_h))
        hand_img = cv2.medianBlur(hand_img, 5)
        # 阈值分割，采用自适应阈值分割法， maxValue的就是就是最大值为255，超过阈值的部分为maxValue，未超过为0,最终结果就是为255的地方为需要留下的手势的部分，而为0的地方则是需要去除的
        mask = cv2.adaptiveThreshold(hand_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresholdType=cv2.THRESH_BINARY, blockSize=blocksize, C=c_num)

        # 1.只把手的地方抠出来，这样做，里面可能会有小的空白
        # 提取出来手的mask,有手的地儿为255，没有手的为0，这样还能把小的连通域去掉
        x = cv2.connectedComponentsWithStats(mask)
        hand_mask1 = np.zeros((new_h, new_w), dtype=np.uint8)  # 未做腐蚀的
        areas = x[2][:, 4]  # 各个连通域的面积
        max_num = np.argmax((areas[1:])) + 1  # 背景的面积是第一个（索引为0），要去除背景之后选择最大的那个才是手的面积，再算上背景的索引，索引号就要+1,也就是说max_num就是手的索引
        for i in range(new_h):
            for j in range(new_w):
                if x[1][i, j] == max_num:  # 如果标签图上的某一个值等于手的索引
                    hand_mask1[i, j] = 255

        gesture_num = int(img_name.split('_')[1])  # 是第几个手势
        if (gesture_num in gesture_num_list_blank_ok) or if_blank_ok:
            hand_mask2 = hand_mask1
        else:
            # 既不在可以存在手内部空白框  而且也不能存在空白
            # 2. 把背景去掉，其余的（包括手和手里面的小空白）都留着
            hand_mask1_1 = np.zeros((new_h + 2, new_w + 2), dtype=np.uint8)  # 为了防止北京场被手分割成2块或者以上，就外面加了一圈，这样背景场（0）可以永远连通
            hand_mask1_1[1:-1, 1:-1] = hand_mask1
            hand_mask1_1 = 255 - hand_mask1_1  # 反过来，手是背景0,其余都是255（背景和手内部的小空白）
            x = cv2.connectedComponentsWithStats(hand_mask1_1)
            areas = x[2][:, 4]  # 各个连通域的面积
            max_num = np.argmax((areas[1:])) + 1  # 找到背景场的编号
            hand_mask2 = np.zeros((new_h, new_w), dtype=np.uint8)  # 未做腐蚀的
            for i in range(new_h):
                for j in range(new_w):
                    if x[1][i + 1, j + 1] != max_num:  # 如果标签图上的某一个值不等于背景场的值
                        hand_mask2[i, j] = 255
        kernel = np.ones((3, 3), dtype=np.uint8)
        hand_mask_erosion = cv2.erode(hand_mask2, kernel, iterations=1)  # 腐蚀以后的mask
        hand_save = np.ones((new_h, new_w)) * 255  # 用来保存抠出来的图
        hand_save = hand_save.astype(np.uint8)
        hand_save_erosion = np.ones((new_h, new_w)) * 255  # 用来保存抠出来并且腐蚀以后的图
        hand_save_erosion = hand_save_erosion.astype(np.uint8)
        for i in range(new_h):
            for j in range(new_w):
                if hand_mask2[i, j] != 0:  # 不等于0就是手
                    if hand_img[i, j] == 255:  # 因为背景就是255，如果原图也是255，那么就会被当成背景
                        hand_save[i, j] = 254
                    else:
                        hand_save[i, j] = hand_img[i, j]
                if hand_mask_erosion[i, j] != 0:  # 不等于0就是手
                    if hand_img[i, j] == 255:  # 因为背景就是255，如果原图也是255，那么就会被当成背景
                        hand_save_erosion[i, j] = 254
                    else:
                        hand_save_erosion[i, j] = hand_img[i, j]
        # hand_all = np.hstack((hand_img, mask, hand_mask1, hand_mask2, hand_mask_erosion, hand_save, hand_save_erosion))
        hand_all = np.hstack((hand_img, hand_save, hand_save_erosion))

        if if_show:
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, 2)
            cv2.imshow('result', img)
            cv2.imshow('hand_all', hand_all)
            print(
                'resize以后的原图  初始mask  第一次连通域处理   第二次联通与处理  腐蚀处理   腐蚀以前的手  腐蚀以后的手')
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        cv2.imwrite(out_img_path + '/' + input_name + '/' + str(c_num) + '/' + img_name, hand_all)
        np.savetxt(out_txt_path + '/' + input_name + '/' + str(c_num) + '/' + img_name[:-3] + 'txt', hand_save_erosion,
                   fmt="%d")

cv2.destroyAllWindows()
