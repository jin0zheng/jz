import cv2

# 调用摄像头
# VideoCapture的参数是指调用电脑那个摄像头，笔记本电脑一般默认为0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取视频
    ret, frame = cap.read()
    if ret:
        if cv2.waitKey(30) == ord('q'):
            break
        # 显示画面
        cv2.imshow('face', frame)
# 释放摄像头
cap.release()
# 销毁窗口
cv2.destroyAllWindows()
