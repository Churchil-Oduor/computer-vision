import cv2 as cv

img = cv.imread("../cat.jpeg")
cv.imshow("img", img)

canny = cv.Canny(img, 125, 175)
cv.imshow("Canny", canny)

cv.waitKey(0)
