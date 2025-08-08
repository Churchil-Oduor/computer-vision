#!/usr/bin/env python3

import cv2 as cv
import mediapipe as mp

class PoseDetector:
    def __init__(self):

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, position=[], draw=True):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, lm)
                    if len(position) == 0:
                        cv.circle(frame, (cx, cy), 10, (255, 0, 8), cv.FILLED)
                    elif id in position:
                        cv.circle(frame, (cx, cy), 10, (255, 0, 8), cv.FILLED)
                        
                    cv.imshow("frame", frame)


