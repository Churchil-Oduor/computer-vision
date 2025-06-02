import cv2 as cv

def is_rectangle(contour):
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    return len(approx) == 4 and cv.isContourConvex(approx), approx
