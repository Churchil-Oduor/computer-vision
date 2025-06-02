import cv2 as cv

img = cv.imread("../cat.jpeg")
cv.imshow("img", img)
# convertion to grayscale
grayed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("grayed", grayed)

cv.waitKey(0)
