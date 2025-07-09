import cv2
import numpy as np

# Loading the image
image_path = 'aruco.png'  
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load the image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Create ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect markers
corners, ids, rejected = detector.detectMarkers(gray)

# Draw detected markers on the image
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    print(f"Detected {len(ids)} ArUco markers with IDs: {ids.flatten()}")
else:
    print("No ArUco markers detected.")

# Display the result
cv2.imshow('Detected ArUco Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
