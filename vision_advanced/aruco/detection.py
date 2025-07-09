#!/usr/bin/env python3

import cv2
import numpy as np

# Load the image
image_path = 'test file.png'  # Ensure this matches your file
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load the image at {image_path}. Check the file path and ensure the file exists.")
    exit()

# Print image details for debugging
print(f"Image shape: {image.shape}, Type: {image.dtype}")
print("Sample pixel values (top-left 2x2):", image[0:2, 0:2])
print("Pixel value range:", np.min(image), "to", np.max(image))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Lenient detection parameters
parameters.minMarkerPerimeterRate = 0.005
parameters.maxMarkerPerimeterRate = 10.0
parameters.polygonalApproxAccuracyRate = 0.15
parameters.adaptiveThreshConstant = 3
parameters.minCornerDistanceRate = 0.01
parameters.minDistanceToBorder = 0
parameters.maxErroneousBitsInBorderRate = 0.8
parameters.errorCorrectionRate = 1.0

# Create ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect markers
corners, ids, rejected = detector.detectMarkers(gray)

# Debugging: Print rejected candidates
print(f"Rejected candidates: {len(rejected)}")
if rejected:
    print("Rejected candidate corners:", rejected)

# Draw detected markers
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    print(f"Detected {len(ids)} ArUco marker(s) with ID(s): {ids.flatten()}")
else:
    print("No ArUco markers detected. Possible reasons:")
  
# Save the grayscale image for debugging
cv2.imwrite('grayscale_marker.png', gray)
print("Grayscale image saved as grayscale_marker.png for debugging")

# Draw rejected candidates in red
if rejected:
    cv2.aruco.drawDetectedMarkers(image, rejected, borderColor=(0, 0, 255))
    print("Rejected candidates drawn in red on output image")

# Save the output image
cv2.imwrite('aruco_detected.png', image)
print("Output image saved as aruco_detected.png")

# Display the result (optional, skip if display fails)
try:
    cv2.imshow('Detected ArUco Markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Display error: {e}. Check output image 'aruco_detected.png' instead.")
