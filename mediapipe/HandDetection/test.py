import cv2 as cv
import mediapipe as mp
cap = cv.VideoCapture(0)

while True:
    isTrue, frame = cap.read()

    cv.imshow("frame", frame)
    cv.waitKey(1)
