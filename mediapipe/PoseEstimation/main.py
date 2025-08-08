#!/usr/bin/env python3

import Pose as ps
import cv2 as cv


def main():
    cap = cv.VideoCapture(0)
    detector = ps.PoseDetector()
    
    while True:
        isTrue, frame = cap.read()
        
        if not isTrue:
            print("Error in frame")
            break

        pose = detector.findPose(frame, position = [4])
        cv.waitKey(1)

if __name__=="__main__":
    main()
