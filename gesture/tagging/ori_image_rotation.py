# 对采集完，未标注的图片进行旋转，增加角度多样性，这个注意一下，手要在图片中心
import os

import albumentations as al
import cv2
from random import randint
input_folder_path = \
    r'D:\jz_script\gesture\capture_hand\BU\img'  # 输入的文件夹名字，里面都是各种图片
os.chdir(input_folder_path)
out_folder_path = '../rotate1'  # 输出的文件夹
if os.path.exists(out_folder_path):
    pass
else:
    os.mkdir(out_folder_path)
s = randint(-180,180)
print(s)
limit = [s,s]  # 旋转范围
limit = [158,158] # 【-90，90】

transform = al.Compose([al.Rotate(limit=limit, p=1)])

name_list = os.listdir(input_folder_path)
name_list.sort()

for name in name_list:
    num = name.split("_")[1]
    image = cv2.imread(name, 0)
    transformed = transform(image=image)
    image = transformed["image"]
    # cv2.imshow('1', image)
    # key = cv2.waitKey()
    cv2.imwrite(out_folder_path + '/' + name, image)
# cv2.destroyAllWindows()
