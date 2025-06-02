import cv2 as cv
from rotation import rotate

capture = cv.VideoCapture(0)

while True:
    frame_check, frame = capture.read()

    if frame_check:
        R_frame = rotate(frame, 180)
        cv.imshow("Rotated", R_frame)
        cv.imshow("actual", frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

