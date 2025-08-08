#!/usr/bin/env python3

import cv2 as cv
import mediapipe as mp

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hand = mpHands.Hands()

cap = cv.VideoCapture(0)


while True:
    isTrue, frame = cap.read()

    if not isTrue:
        print("Error in frame")
        break

    flipped = cv.flip(frame, 1)
    imgrgb = cv.cvtColor(flipped, cv.COLOR_BGR2RGB)
    results = hand.process(imgrgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = flipped.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 13:
                    cv.circle(flipped, (cx, cy), 15, (255, 0, 255), cv.FILLED)
           
            mpDraw.draw_landmarks(flipped, handLms, mpHands.HAND_CONNECTIONS)


    cv.imshow("image", flipped)
    
    cv.waitKey(1)

