import cv2
import numpy as np
import urllib.request
import time

# MJPEG URL from IP Webcam (replace with your phone's IP and port)
mjpeg_url = "http://192.168.43.68:8080"  # Update with your IP Webcam URL

# Initialize video capture with the MJPEG stream
cap = cv2.VideoCapture(mjpeg_url)

if not cap.isOpened():
    print(f"Error: Could not open MJPEG stream at {mjpeg_url}. Ensure IP Webcam is running, using the front camera, both devices are on the same Wi-Fi, and no firewall is blocking port 8080.")
    exit()

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.minMarkerPerimeterRate = 0.005
parameters.maxMarkerPerimeterRate = 10.0
parameters.polygonalApproxAccuracyRate = 0.15
parameters.adaptiveThreshConstant = 3
parameters.minCornerDistanceRate = 0.01
parameters.minDistanceToBorder = 0
parameters.maxErroneousBitsInBorderRate = 0.8
parameters.errorCorrectionRate = 1.0
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

print("Starting live ArUco marker detection from MJPEG stream (front camera, local Wi-Fi). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from MJPEG stream. Check the connection or Wi-Fi.")
        time.sleep(1)
        cap = cv2.VideoCapture(mjpeg_url)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"Detected {len(ids)} ArUco marker(s) with ID(s): {ids.flatten()}")
    else:
        print("No ArUco markers detected in this frame.")

    if rejected:
        cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))
        print(f"Rejected candidates: {len(rejected)}")

    cv2.imshow('Live ArUco Detection (MJPEG)', frame)
    cv2.imwrite('mjpeg_frame.png', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("MJPEG stream terminated.")
