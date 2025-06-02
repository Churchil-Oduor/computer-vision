import cv2 as cv

img = cv.imread("../cat.jpeg")
cv.imshow("img", img)

# blurring using the gaussian blur
blurred = cv.GaussianBlur(img, (9, 9), cv.BORDER_DEFAULT)
cv.imshow("blurred", blurred)

cv.waitKey(0)


