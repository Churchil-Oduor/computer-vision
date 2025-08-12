import cv2
import numpy as np

def draw_sphere(frame, center, radius, color=(0, 0, 255)):
    """Draws a shaded sphere (fake 3D) on the frame."""
    sphere = np.zeros((radius*2, radius*2, 3), dtype=np.uint8)
    for y in range(radius*2):
        for x in range(radius*2):
            dx = x - radius
            dy = y - radius
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radius:
                # Simulate lighting by shading
                shade = 1 - (dist / radius)
                b = int(color[0] * shade)
                g = int(color[1] * shade)
                r = int(color[2] * shade)
                sphere[y, x] = (b, g, r)
    # Overlay sphere on frame
    x_start = max(center[0] - radius, 0)
    y_start = max(center[1] - radius, 0)
    x_end = min(center[0] + radius, frame.shape[1])
    y_end = min(center[1] + radius, frame.shape[0])

    sphere_x_start = max(0, radius - center[0])
    sphere_y_start = max(0, radius - center[1])
    sphere_x_end = sphere_x_start + (x_end - x_start)
    sphere_y_end = sphere_y_start + (y_end - y_start)

    roi = frame[y_start:y_end, x_start:x_end]
    sphere_part = sphere[sphere_y_start:sphere_y_end, sphere_x_start:sphere_x_end]

    mask = cv2.cvtColor(sphere_part, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    sphere_fg = cv2.bitwise_and(sphere_part, sphere_part, mask=mask)
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    combined = cv2.add(bg, sphere_fg)

    frame[y_start:y_end, x_start:x_end] = combined

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw a fake 3D sphere at center of screen
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    draw_sphere(frame, (center_x, center_y), 100, (0, 0, 255))  # red sphere
    
    cv2.imshow("Live Feed with Sphere", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

