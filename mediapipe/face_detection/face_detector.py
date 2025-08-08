#!/usr/bin/env python3


import face_detector as fc
import cv2 as cv
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_detectionCon=0.5):
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFace.FaceDetection(min_detectionCon)


    def findFace(self, frame, draw=True):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face.process(rgb)
        
        bboxs = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h) 
                bboxs.append([id, bbox, detection.score])
                if draw:
                    frame = self.fancyDraw(frame, bbox)
                    cv.rectangle(frame, bbox, (255, 0, 8), 2)
                cv.imshow("frame", frame)

    def fancyDraw(self, frame, bbox, l=30, t=10, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv.rectangle(frame, bbox, (255, 0, 255), rt)
        # top left
        cv.line(frame, (x, y), (x+l , y), (255, 0, 255), t)
        cv.line(frame, (x, y), (x, y+l), (255, 0, 255), t)

        #top right
        cv.line(frame, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(frame, (x1, y), (x1, y+l), (255, 0, 255), t)

        #bottom left
        cv.line(frame, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(frame, (x, y1), (x, y1 - l) ,(255, 0, 255), t)

        #bottom right
        cv.line(frame, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(frame, (x1, y1), (x1, y1 - l), (255, 0, 255), t)



        return frame
