import cv2 as cv

img = cv.imread("../cat.jpeg")
cv.imshow("img", img)
dilated = cv.dilate(img, (7, 7), iterations=1)

cv.imshow("dilated", dilated)

cv.waitKey(0)
