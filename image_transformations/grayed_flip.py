import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    frame_check, frame = capture.read()

    if frame_check:
        F_frame = cv.flip(frame, 1) # 0 -> 1 to get horizontal flip
        grayed = cv.cvtColor(F_frame, cv.COLOR_BGR2HSV)
        cv.imshow("FLipped", grayed)
        cv.imshow("actual", frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

