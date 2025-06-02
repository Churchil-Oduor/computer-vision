import cv2 as cv
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define virtual key rectangles (x1, y1, x2, y2)
key_regions = [
    ((100, 200), (200, 300)),  # Key 1
    ((210, 200), (310, 300)),  # Key 2
    ((320, 200), (420, 300))   # Key 3
]

def is_inside(rect, point):
    (x1, y1), (x2, y2) = rect
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Draw the key regions
    for i, ((x1, y1), (x2, y2)) in enumerate(key_regions):
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, f"Key {i+1}", (x1+10, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip
            index_tip = handLms.landmark[8]
            x_tip = int(index_tip.x * w)
            y_tip = int(index_tip.y * h)

            cv.circle(frame, (x_tip, y_tip), 8, (255, 0, 0), -1)

            for i, region in enumerate(key_regions):
                if is_inside(region, (x_tip, y_tip)):
                    cv.putText(frame, f"Key {i+1} Pressed", (10, 50 + i * 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv.rectangle(frame, region[0], region[1], (0, 0, 255), 3)

    cv.imshow("Virtual Piano", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

