import face_detector as fd
import cv2 as cv


cap = cv.VideoCapture(0)

while True:
    isTrue, frame = cap.read()

    if not isTrue:
        print("Error in Frame")
        break

    detector = fd.FaceDetector()
    detector.findFace(frame, draw=True)

    cv.waitKey(1)

