#!/usr/bin/env python

import cv2 as cv

img = cv.imread("photos/CAT.jpeg")

cv.imshow("CAT", img)

cv.waitKey(0)
