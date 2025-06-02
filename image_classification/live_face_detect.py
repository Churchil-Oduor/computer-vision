#!/usr/bin/env python3
import cv2 as cv

capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    frame_check, frame = capture.read()

    if frame_check:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # passing the frame into the classifier
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=-1)
        for (x, y, a, b) in faces_rect:
            cv.rectangle(frame, (x, y), (x+a, y+b), (0, 255, 0), thickness=2)

        cv.imshow("face detection", frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
