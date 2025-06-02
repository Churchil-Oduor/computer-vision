import cv2 as cv

vd = cv.VideoCapture(0)
count_frame = 0

while True:
    frame_check, frame = vd.read()

    if frame_check:
        cv.imshow("video", frame)

        if 0xff == ord('d') and cv.waitKey(20):
            break

vd.release()
cv.destroyAllWindows()
