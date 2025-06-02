#!/usr/bin/env python3

import cv2 as cv
from rotation import rotate

img = cv.imread("../cat.jpeg")

rotated = rotate(img, -45)
cv.imshow("rotated", rotated)

cv.waitKey(0)



