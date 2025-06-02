#!/usr/bin/env python3
import cv2 as cv

img = cv.imread("cat.jpeg")
print(type(img))
cv.imshow("Cat", img)

cv.waitKey(0)

