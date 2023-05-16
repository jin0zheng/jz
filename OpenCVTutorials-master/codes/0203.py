import numpy as np
import cv2 as cv
# 创建黑色的图像
img = np.full((480,480,3),255, np.uint8)
# 绘制一条厚度为5的蓝色对角线
cv.line(img,(0,0),(480,480),(255,0,0),5)
# 绘制一条厚度为五的绿色对角线
cv.line(img,(0,480),(480,0),(0,255,0),5)
# 绘制矩阵
cv.rectangle(img,(220,220),(260,260),(0,0,255),0)
# 绘制圆
cv.circle(img,(240,240),60,(0,0,0))
# 绘制椭圆
cv.ellipse(img,(240,240),(60,30),0,0,-360,225,-1)
# 绘制文本
cv.putText(img, 'out_txt', (40, 40), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

cv.namedWindow('1',cv.WINDOW_AUTOSIZE)
cv.imshow('1',img)
cv.waitKey()