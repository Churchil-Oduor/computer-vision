import cv2 as cv
import utils as util

def resizeFrame(frame, resize=0.75):
    width = int(frame.shape[1] * resize)
    height = int(frame.shape[0] * resize)
    dimensions = (width, height)
    new_capture_obj = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    return new_capture_obj



capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()

    resized_frame = resizeFrame(frame, 1.2)
    cv.imshow("video", resized_frame)

    if cv.waitKey(20) and 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()


