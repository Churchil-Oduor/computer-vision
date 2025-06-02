#!/usr/bin/env python3
import cv2 as cv

capture = cv.VideoCapture(0)
scale = 0.9
while True:
    isTrue, frame = capture.read()
    
    if isTrue:
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (height, width)
        frameX = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

        ## canny the frame
        canny_frame = cv.Canny(frameX, 125, 175)
        cv.imshow("image", canny_frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
