#!/usr/bin/env python3

import cv2 as cv
import mediapipe as mp

class HandDetector:

    def __init__(self, mode=False, maxHands=2, min_detectionCon=0.5,
                 min_trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.min_detectionCon = min_detectionCon
        self.min_trackingCon = min_trackingCon
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode,  self.maxHands, min_detection_confidence=min_detectionCon, min_tracking_confidence=min_trackingCon)

    def findHands(self, img, draw=True):

        """
        find and tracks the hand.

        img: image frame of the hand
        draw: bool to enable drawing landmarks over the hand in realtime

        return: returns the hand found

        """

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, handNo=0, marks=[], draw=True):

        """
        img: image found
        handNo: index of hand.
        marks: which landmark to detect...range from 0 to 20
        draw:  bool to enable drawing over the position found in real time.
        return: List of landmarks found

        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        landmarks = results.multi_hand_landmarks

        lmList = []
        if landmarks:
            myHand = landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and id in marks:
                    cv.circle(img, (cx, cy), 8, (255, 0, 8), cv.FILLED)
        return lmList

