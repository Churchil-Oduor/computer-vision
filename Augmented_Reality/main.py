#!/usr/bin/env python3
#import engine as eng
import engine_test as eng
import cv2 as cv

def main():

    cap = cv.VideoCapture(0)
    handDetector = eng.HandTrack()

    while True:
        isTrue, frame = cap.read()

        if not isTrue:
            print("Error in feed")
            break


        frame = cv.flip(frame, 1)
        frame, gestures = handDetector.handPositions(frame, True)
        
        h, w, _ = frame.shape

        #####
        cx, cy = w // 2, h // 2
        cv.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)
        cv.line(frame, (0, cy), (w, cy), (0, 255, 0), 2)
        #####



        cv.imshow("frame", frame)
        cv.waitKey(1)

if __name__=="__main__":
    main()
