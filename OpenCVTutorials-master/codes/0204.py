import numpy as np
import cv2 as cv

drawing = False  # 如果按下鼠标，则为真
mode = True  # 如果为真，绘制矩形。按 m 键可以切换到曲线
ix, iy = -1, -1


# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -0)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -0)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 50, (0, 0, 255), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)
while (1):
    cv.imshow('image', img)
    if cv.waitKey(2) & 0xFF == 27:
        break
cv.destroyAllWindows()