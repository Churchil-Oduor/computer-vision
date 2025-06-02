#!/usr/bin/env python3
import cv2 as cv
from is_rectangle import is_rectangle as is_rec

feed = cv.VideoCapture(0)

while True:
    frame_check, frame = feed.read()

    if not frame_check:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 23, 120)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        is_rect, approx = is_rec(cnt)

        if is_rect:
            cv.drawContours(frame, [approx], -1, (0, 255, 0), 3)
    cv.imshow("rectangle_detected", frame)


    if cv.waitKey(1) & 0xff == ord('d'):
        break

feed.release()
cv.destroyAllWindows()
