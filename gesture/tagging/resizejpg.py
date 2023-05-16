

# path = "D:\jz_script\yolov7-pytorch-master\VOCdevkit\VOC2007\img"
# os.chdir(path)
#
# new_image = Image.new('RGB', (864, 800), (225, 255, 255))
# # 在新图像中粘贴原始图像
# n = 0
# for i in os.listdir(path):
#     im = Image.open(i)
#     width, height = im.size
#     new_image.paste(im, (0, 0))
#     new_image.save(i)
#     if n % 1000 == 0:
#         print(n // 1000)
#     n += 1
from PIL import Image
import os
path = r"D:\jz_script\gesture\capture_hand\xiangkangyi\img"
os.chdir(path)

for i in os.listdir(path):
    new_img = Image.new('RGB', (848, 450), (0,0,0))
    im = Image.open(i)
    print(im.size)
    break
    # im.show()
    im.paste(new_img,(0,350))
    im.save(i)
