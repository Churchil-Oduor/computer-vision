#!/usr/bin/env python3

import cv2 as cv
import mediapipe as mp
import numpy as np


cap = cv.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()

while True:
    isTrue, img = cap.read()
    
    if not isTrue:
        print("Encountered Error in frame!")
        break

    imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    flipped = cv.flip(imgrgb, 1)    
    cv.imshow("image", flipped)
    results = hands.process(flipped)
    print(results.multi_hand_landmarks)
    cv.waitKey(1)

