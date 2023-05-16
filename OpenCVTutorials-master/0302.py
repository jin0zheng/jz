import cv2 as cv
img = cv.imread('img_1.png')
cv.imshow('1',img)
cv.waitKey()
img = img + img
cv.imshow('1',img)
cv.waitKey()
dst = cv.addWeighted(img,0.7,img,0.3,0)
cv.imshow('dst',dst)
cv.waitKey(0)