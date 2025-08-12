import cv2 as cv
import aruco as arc
import numpy as np
import render as r


cap = cv.VideoCapture(0)
arc = arc.Aruco()
rend = r.Render(alpha=0.5)
marker_length = 0.05
frame_width, frame_height = 640, 480
focal_length = frame_width
camera_matrix = np.array([[focal_length, 0, frame_width / 2],
                         [0, focal_length, frame_height / 2],
                         [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))

reference_distance = 0.5
scale_reference = 1.0
png = cv.imread("pin1.png", cv.IMREAD_UNCHANGED)

while True:
    isTrue, frame = cap.read()

    if not isTrue:
        print("Error in frame")
        exit()

    ids, corners, frame = arc.detection(frame)
    if ids is not None:
        rvecs, tvecs, _objpnts = cv.aruco.estimatePoseSingleMarkers(corners,
                                                                 marker_length, camera_matrix, 
                                                                 dist_coeffs)
        for rvec, tvec, corner in zip(rvecs, tvecs, corners):
            cv.drawFrameAxes(frame, 
                           camera_matrix, dist_coeffs,
                           rvec, tvec, 0.07)

            current_distance = rend.lowPassFilter(tvec[0][2])
            center = (frame.shape[0] // 2, frame.shape[1] // 2)
#            cx, cy = int(center[0]), int(center[1])
            scale = scale_reference * (reference_distance / current_distance)
            rend.overlayImg(frame, png, center, scale)
            print(f"Transvec {tvec}")
    frame = cv.flip(frame, 1)
    cv.imshow("frame", frame)
    cv.waitKey(1)
