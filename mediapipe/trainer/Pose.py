#!/usr/bin/env python3

import cv2 as cv
import mediapipe as mp
import math


class PoseDetector:
    def __init__(self):

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, position=[], draw=True):
        """

        Finds all the positional landmarks of the body

        frame: image to be processed.
        position: positions to mark.
        draw: put the marker or not on the position selected.

        Returns all the position landmarks in the feed
        """
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        positions = []
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    positions.append([id, cx, cy])
                    if id in position:
                        cv.circle(frame, (cx, cy), 10, (255, 0, 8), cv.FILLED)
                    cv.imshow("frame", frame)
                return positions
            else:
                return positions

    def markPose(self, frame, p1, p2, p3):
        #shoulder
        cv.circle(frame, p1, 10, (255, 0, 0), 10)

        # elbow
        cv.circle(frame, p2, 10, (255, 0, 0), 10)

        # wrist
        cv.circle(frame, p3, 10, (255, 0, 0), 10)

        # shoulder to elbow : roughly constant
        cv.line(frame, p1, p2, (255, 255, 255), 10)

        #elbow to wrist: roughly constant
        cv.line(frame, p2, p3, (255, 255, 255), 10)
        angle = self.findAngle(p1, p2, p3)
        cv.putText(frame, f"{angle}", (p2[0]+10, p2[1]+10), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 5)
        return frame

    def findAngle(self, p1, p2, p3, draw=True):
        """
        Calculates angle using the cosine rule.
        p1: shoulder coordinate
        p2: elbow coordinate
        p3: wrist coordinate

        returns: angle extended at the elbow
        """
        line_1 = self.lineLengthCalculator(p1, p2) # shoulder to elbow, constant
        line_2 =self.lineLengthCalculator(p2, p3)  # elbow to wrist, constant
        variable_line = self.lineLengthCalculator(p1, p3) # shoulder to wrist, variable
#        print(f"1. {line_1}\n 2. {line_2}\n 3. {variable_line}")
        cosine = ((line_1 ** 2 + line_2 ** 2) - variable_line ** 2) / (2 * line_1 * line_2)
        self.angle = int(math.degrees(math.acos(cosine)))


        print(f"line 1: {line_1}\nline 2: {line_2}\nline 3: {variable_line}\nAngle: {self.angle}\n\n\n")

        return self.angle

    def lineLengthCalculator(self, point_1=(0, 0), point_2=(0, 0)):

        """
        computes the length of the joints
        point_1: x, y cordinate of point 1
        point_2: x, y cordinate of point 2
        
        return: length between point 1 and point 2
        """

        y_difference = point_1[1] - point_2[1]
        x_difference = point_1[0] - point_2[0]
        square_sum = (y_difference ** 2) + (x_difference ** 2)
        distance = math.sqrt(square_sum)
        return distance

