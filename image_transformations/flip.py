import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    frame_check, frame = capture.read()

    if frame_check:
        F_frame = cv.flip(frame, 0) # 0 -> 1 to get horizontal flip
        cv.imshow("FLipped", F_frame)
        cv.imshow("actual", frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

