#!/usr/bin/env python3

import cv2 as cv
import mediapipe as mp
import numpy as np


cap = cv.VideoCapture(0)


while True:
    isTrue, img = cap.read()
    
    if not isTrue:
        print("Encountered Error in frame!")
        break

    flipped = cv.flip(img, 1)
    cv.imshow("image", flipped)
    cv.waitKey(1)

