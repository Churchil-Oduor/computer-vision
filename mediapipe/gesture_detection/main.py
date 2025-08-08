import cv2 as cv
import os
import handDetector as htm

cap = cv.VideoCapture(0)
wCam, hCam = 640, 480

cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector(min_detectionCon = 0.7)
lmlist = []
tips = [4, 8, 12, 16, 20]
fingers = [0, 0, 0, 0, 0]

while True:

    success, frame = cap.read()

    if not success:
        print("Error in frame")
        break

    frame = cv.flip(frame, 1)
    lmlist = detector.findPositions(frame)

    if len(lmlist) != 0:
        yindex = 2
        for i in range(5):
            if i > 0:
                if lmlist[tips[i]][yindex] < lmlist[tips[i] - 1][yindex]:
                    fingers[i] = 1
                else:
                    fingers[i] = 0
            else:
                xindex = 1
                if lmlist[tips[i]][xindex] < lmlist[tips[i] - 1][xindex]:
                    fingers[i] = 1
                else:
                    fingers[0] = 0
                finger_count = fingers.count(1)
            cv.rectangle(frame, (20, 225), (178, 425), (170, 425), cv.FILLED)
            cv.putText(frame, str(finger_count), (45, 375), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 8), 10)
    cv.imshow("frame", frame)
    cv.waitKey(1)
