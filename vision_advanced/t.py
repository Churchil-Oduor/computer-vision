import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)

        if len(approx) == 4 and area > 10000:
            # Draw the rectangle
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

            # Get bounding box and compute center
            x, y, w, h = cv2.boundingRect(approx)
            cx, cy = x + w // 2, y + h // 2
            dot_size = 20  # Size of square region over the "O"

            # Extract region at center
            dot_roi = gray[cy - dot_size:cy + dot_size, cx - dot_size:cx + dot_size]
            avg_brightness = np.mean(dot_roi)

            # Show detection
            color = (0, 255, 0) if avg_brightness < 100 else (0, 0, 255)
            status = "O COVERED" if avg_brightness < 100 else "O VISIBLE"

            cv2.rectangle(frame, (cx - dot_size, cy - dot_size), (cx + dot_size, cy + dot_size), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Detect Obstruction", edged)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

