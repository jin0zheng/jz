import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

# 读取图像
img = cv.imread('1.jpg', 0)
# cv.WINDOW_AUTOSIZE  创建空窗口，并直接显示
cv.namedWindow('image', cv.WINDOW_NORMAL)
# 在前面创建的窗口展示图片
cv.imshow('image', img)
# 键盘绑定函数，单位ms，默认0一直等待
key = cv.waitKey(0) & 0xFF
if key == ord('s'):
    cv.imwrite('00.jpg', img)

elif key == 27:  # esc
    cv.destroyAllWindows()
#
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # 隐藏 x 轴和 y 轴上的刻度值
plt.show()
