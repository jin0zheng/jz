import numpy as np
import cv2 as cv

# 从摄像头抓捕视频
cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc,20.0,(640,480))
if cap.isOpened():
    print(cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT)
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('frame', gray)
            out.write(frame)
            if cv.waitKey(10) == ord('q'):
                break
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
else:
    print("Cannot open camera")
out.release()
cap.release()
cv.destroyAllWindows()
