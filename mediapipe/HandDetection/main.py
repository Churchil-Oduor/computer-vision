#!/usr/bin/env python3

import cv2 as cv
import handDetector as hd

def main():

    cap = cv.VideoCapture(0)
    handDetector = hd.HandDetector()
    pos = []
    while True:
        isTrue, frame = cap.read()

        if not isTrue:
            print("Error in Frame")
            break

        frame = handDetector.findHand(frame, draw=True)
        #pos = handDetector.findPositions(frame, marks=[4, 8])
        cv.imshow("frame", cv.flip(frame, 1))
        cv.waitKey(1)

if __name__ == "__main__":
    main()
