import cv2
import numpy as np

# Generate sphere vertices (only once)
def generate_sphere(radius=50, resolution=30):
    phi = np.linspace(0, np.pi, resolution)       # polar angle
    theta = np.linspace(0, 2 * np.pi, resolution) # azimuth angle
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    return points

# Projection from 3D to 2D (orthographic)
def project(points, center=(320, 240)):
    return np.column_stack((points[:, 0] + center[0],
                            points[:, 1] + center[1])).astype(int)

# Precompute sphere mesh
sphere_points = generate_sphere(radius=60, resolution=30)

# Live webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Move sphere to (cx, cy)
    cx, cy = 320, 240
    projected = project(sphere_points, center=(cx, cy))

    # Draw solid sphere by filling convex hull
    cv2.fillConvexPoly(frame, projected, (0, 0, 255))

    cv2.imshow("Sphere Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

