import cv2 as cv
import numpy as np
import Pose as ps


cap = cv.VideoCapture(0)
detector = ps.PoseDetector()
poselist = []
while True:
    isTrue, frame = cap.read()

    if not isTrue:
        print("Error in frame")
        break

    frame = cv.flip(frame, 1)
    poselist = detector.findPose(frame, draw=True, position=[11])
    if poselist is None:
        cv.imshow("frame", frame)
    else:
        _, x1, y1 = poselist[11]
        _, x2, y2 = poselist[13]
        _, x3, y3 = poselist[15]
        p1 = (x1, y1)
        p2 = (x2, y2)
        p3 = (x3, y3)
        frame = detector.markPose(frame, p1, p2, p3)
        cv.imshow("frame", frame)
    cv.waitKey(1)
