import cv2 as cv
import utils as util

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    resized_frame = util.video_resize(frame, 0.5)

    cv.imshow("video", resized_frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

