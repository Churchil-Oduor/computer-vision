import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Draw all contours in light blue for debugging
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 200, 200), 1)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)

        # Only consider big quadrilaterals as rectangles
        if len(approx) == 4 and area > 10000:
            # Draw polygon approx in green
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # bounding box

            cx, cy = x + w // 2, y + h // 2
            dot_size = 20
            # Ensure ROI doesn't go out of frame bounds
            y1, y2 = max(0, cy - dot_size), min(gray.shape[0], cy + dot_size)
            x1, x2 = max(0, cx - dot_size), min(gray.shape[1], cx + dot_size)
            dot_roi = gray[y1:y2, x1:x2]

            avg_brightness = np.mean(dot_roi)

            # Show the ROI rectangle around the dot
            color = (0, 255, 0) if avg_brightness < 100 else (0, 0, 255)
            status = "O COVERED" if avg_brightness >= 100 else "O VISIBLE"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{status} {int(avg_brightness)}", (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Detect Obstruction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

