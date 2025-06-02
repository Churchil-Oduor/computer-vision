import cv2 as cv

def resize(frame, scale=0.9):
    width = int(frame.shape[0] * scale)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture("videos/karate.mp4")

while True:
    isTrue, frame = capture.read()
    frame_resized = resize(frame)

    cv.imshow('video', frame_resized)
    if cv.waitKey(10) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
