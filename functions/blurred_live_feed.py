#!/usr/bin/env python3
import cv2 as cv

capture = cv.VideoCapture(0)
scale = 1
while True:
    isTrue, frame = capture.read()
    
    if isTrue:
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (height, width)
        frameX = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

        ## blur the frame
        blurred_frame = cv.GaussianBlur(frameX, (7,7), cv.BORDER_DEFAULT)
        cv.imshow("image", blurred_frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
