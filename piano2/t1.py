import cv2
import pytesseract

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional: threshold to improve recognition
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Extract text
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    boxes = pytesseract.image_to_boxes(thresh)

    # Draw bounding boxes
    h, w, _ = frame.shape
    for b in boxes.splitlines():
        b = b.split()
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(frame, (x, h - y), (x2, h - y2), (0, 255, 0), 2)
        cv2.putText(frame, b[0], (x, h - y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("OCR Letter Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

