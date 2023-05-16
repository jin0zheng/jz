import cv2
import os
import numpy as np

path = r'D:\script\gesture\capture_hand\wuchi2'
os.chdir(path)
out_path = "../img2"
if not os.path.exists(out_path):
    os.mkdir(out_path)
# filp_num = [0, 7, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28]
# filp_num = [3]3
# filp_num = [str(i) for i in filp_num]
for i in os.listdir(path):

    s = cv2.imread(i)
    cv2.imwrite("../img2/%s" % i, img=np.flip(s, axis=1))
