import numpy as np
import cv2 as cv
import colorsys


def nothing(x):
    pass


# 创建一个黑色的图像，一个窗口
def mode_change(x):
    if x == 0:
        r, g, b = 0, 0, 0
    elif x == 1:
        r, g, b = 0, 0, 0
    elif x == 2:
        r, g, b = 0, 0, 0
    cv.setTrackbarPos('R', 'image', r)
    cv.setTrackbarPos('G', 'image', g)
    cv.setTrackbarPos('B', 'image', b)


img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('image')
# 创建颜色变化的轨迹栏
cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)
cv.createTrackbar('mode', 'image', 0, 3, mode_change)
# 为 ON/OFF 功能创建开关
switch = '0 : OFF \t1 : ON'
cv.createTrackbar(switch, 'image', 0, 1, nothing)
while (1):
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # 得到四条轨迹的当前位置
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    s = cv.getTrackbarPos(switch, 'image')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]
cv.destroyAllWindows()
