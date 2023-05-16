# 对采集完，未标注的图片进行旋转，增加角度多样性，这个注意一下，手要在图片中心
import os
import albumentations as al
import cv2

input_folder_path = \
    r'D:\jz_script\gesture\capture_hand\jinzheng1\img'  # 输入的文件夹名字，里面都是各种图片
os.chdir(input_folder_path)
out_folder_path = '../rotate1'  # 输出的文件夹
if os.path.exists(out_folder_path):
    pass
else:
    os.mkdir(out_folder_path)
# limit = (85, 85)  # 旋转范围
limit = [-90,-90]  # 【-90，90】

transform = al.Compose([al.Rotate(limit=limit, p=1)])

name_list = os.listdir(input_folder_path)
name_list.sort()

for name in name_list:
    image = cv2.imread(name, 0)
    transformed = transform(image=image)
    image = transformed["image"]
    # cv2.imshow('1', image)
    # key = cv2.waitKey(0)
    cv2.imwrite(out_folder_path + '/' + name, image)
# cv2.destroyAllWindows()
