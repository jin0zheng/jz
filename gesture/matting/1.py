import cv2

# img = cv2.imread("11.png", flags=0)
# s = cv2.adaptiveThreshold(src=img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
#                           thresholdType=cv2.THRESH_BINARY, blockSize=181, C=30, dst=None)
# a = cv2.connectedComponentsWithStats(s)
# # print(a.type)
# cv2.imshow("二值化处理",s)
# cv2.waitKey()

import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype='int8')

B = np.array([[2, 3, 4],
              ], dtype='int8')

C = A + B
print(C)
