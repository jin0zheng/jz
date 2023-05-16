print('16:{:02x}\t8:{:02o}\t2:{:02b}'.format(10, 10, 10))
print(0x0a)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../img.png')
print(img[1:3, 2, 0])
print(img.item(10, 10, 2))
print(img.shape)

img = cv.imread('../img.png', 0)
print(img[1, 1])
print(img.shape)
# cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)
# cv.waitKey(100)
print(img.dtype)

img = cv.imread('../img.png', 1)
r, g, b = cv.split(img)
print(len(r), r.size, len(img))
# cv.merge()

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv.imread('../img_1.png')
replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()