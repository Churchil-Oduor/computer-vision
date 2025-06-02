import cv2 as cv
import utils as util



img = cv.imread("photos/CAT.jpeg")

resized_img = util.img_resize(img, 10)

cv.imshow('cat', resized_img)

cv.waitKey(0)
