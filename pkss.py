import cv2 as cv



cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow("PKSS", frame)

    if cv.waitKey(2) & 0xff == ord('d'):
        break
cv.release()
cap.destroyAllWindows()
