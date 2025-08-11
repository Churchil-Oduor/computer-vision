import cv2
import numpy as np

# Load image
#img = cv2.imread("scene.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)

while True:
    # Apply edge detection
    isTrue, img = cap.read()

    if not isTrue:
        print("Error in frame")
        break

    edges = cv2.Canny(img, 50, 150)
    # Compute image gradients (intensity change)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)  # radians
    # Get the dominant gradient direction
    hist, bins = np.histogram(direction, bins=180, range=(-np.pi, np.pi), weights=magnitude)
    dominant_angle = bins[np.argmax(hist)]
    print(f"Estimated dominant light direction angle: {np.degrees(dominant_angle):.2f}°")
    cv2.imshow("img", img)
    cv2.waitKey(1)

