#!/usr/bin/env python3

import cv2 as cv
import numpy as np

# Load the ArUco dictionary
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# Generate a marker with ID=1, size=400x400 pixels
marker_id = 8
marker_size = 70

# Generate the marker (no input array, let OpenCV handle it)
marker_image = cv.aruco.generateImageMarker(dictionary, marker_id, marker_size)

# Add a white border (standard for ArUco markers)
border_size = marker_size // 7  # Approx 1/7th for border
dimensions = marker_size + 2 * border_size
marker_with_border = np.ones((dimensions, dimensions), dtype=np.uint8) * 255
marker_with_border[border_size:border_size + marker_size, border_size:border_size + marker_size] = marker_image

# Save the marker
cv.imwrite(f"marker_{marker_id}.png", marker_with_border)
print(f"Marker with ID {marker_id} saved as marker_{marker_id}.png (size: {marker_with_border.shape})")
