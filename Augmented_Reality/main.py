#!/usr/bin/env python3
#import engine as eng
import engine_test as eng
import cv2 as cv
import utils as utl
import aruco as arc

def main():

    cap = cv.VideoCapture(0)
    handDetector = eng.HandTrack()
    calc = utl.Calculations()
    aruco = arc.Aruco()

    while True:
        isTrue, frame = cap.read()

        if not isTrue:
            print("Error in feed")
            break

        frame = aruco.detection(frame)
        """

        frame = cv.flip(frame, 1)
        frame, gestures = handDetector.handPositions(frame, True)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        center = calc.center_point((0, cy), (w, cy))
        angle = handDetector.handStatus(frame)
        x1_rot, y1_rot = calc.rotation(center, (0, cy), angle)
        x2_rot, y2_rot = calc.rotation(center, (w, cy), angle)
        cv.line(frame, (int(x1_rot), int(y1_rot)), (int(x2_rot), int(y2_rot)), (0, 255, 0), 2)
        """
        cv.imshow("frame", frame)
        cv.waitKey(1)

if __name__=="__main__":
    main()
