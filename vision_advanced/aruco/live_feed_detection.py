import cv2
import numpy as np

# Initialize the webcam (0 is usually the default camera)
video_url = "https://192.168.43.176:8080/video"
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set camera resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Use lenient detection parameters (based on previous tuning)

parameters.minMarkerPerimeterRate = 0.05 #0.005

parameters.maxMarkerPerimeterRate = 8.0

parameters.polygonalApproxAccuracyRate = 0.15 # 0.15
parameters.adaptiveThreshConstant = 3 # 3
parameters.minCornerDistanceRate = 0.01
parameters.minDistanceToBorder = 0
parameters.maxErroneousBitsInBorderRate = 0.8
parameters.errorCorrectionRate = 1.0

# Create ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

print("Starting live ArUco marker detection. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # Draw detected markers on the frame
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"Detected {len(ids)} ArUco marker(s) with ID(s): {ids.flatten()}")
    else:
        pass
        #print("No ArUco markers detected in this frame.")

    # Draw rejected candidates in red (for debugging, optional)
    if rejected:
        cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))
        print(f"Rejected candidates: {len(rejected)}")

    # Display the frame
    frame = cv2.flip(frame, 1)
    cv2.imshow('Live ArUco Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
print("Live feed terminated.")
