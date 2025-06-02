import cv2 as cv

capture = cv.VideoCapture()

while True:
    isTrue, frame = capture.read()
    cv.imshow('camera', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
