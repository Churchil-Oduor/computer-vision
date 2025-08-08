import cv2 as cv
import numpy as np
import handDetector as htm


wCam, hCam = 1080, 720

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


detector = htm.HandDetector(min_detectionCon=0.7)

while True:
    isTrue, frame = cap.read()

    if not isTrue:
        print("Error in frame")
        break

    frame = cv.flip(frame, 1)
    frame = detector.findHands(frame)
    lmlist = detector.findPositions(frame, draw=False)
    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv.circle(frame, (x1, y1), 10, (255, 0, 4), cv.FILLED)
        cv.circle(frame, (x2, y2), 10, (255, 0, 4), cv.FILLED)
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 10), 3)
        cv.circle(frame, (cx, cy), 10, (255, 0, 4), cv.FILLED)
    cv.imshow("frame", frame)
    cv.waitKey(1)
