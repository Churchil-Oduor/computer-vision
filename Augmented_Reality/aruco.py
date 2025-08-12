import cv2


class Aruco:
    """
    Generates aruco object and initializes it
    """
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.minMarkerPerimeterRate = 0.05 #0.005
        self.parameters.maxMarkerPerimeterRate = 8.0
        self.parameters.polygonalApproxAccuracyRate = 0.15 # 0.15
        self.parameters.adaptiveThreshConstant = 3 # 3
        self.parameters.minCornerDistanceRate = 0.01
        self.parameters.minDistanceToBorder = 0
        self.parameters.maxErroneousBitsInBorderRate = 0.8
        self.parameters.errorCorrectionRate = 1.0
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def detection(self, frame):
        """
        detects Aruco marker
        Args:
            frame(numpy array of image): frame passed for detection
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray_frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"Detected {len(ids)} ArUco marker(s) with ID(s): {ids.flatten()}")

        if rejected:
            cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))
            print(f"Rejected candidates: {len(rejected)}")

        return frame
        
