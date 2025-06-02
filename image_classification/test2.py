import cv2 as cv
import mediapipe as mp

class RectangleDetector:
    def __init__(self):
        pass

    def is_rectangle(self, contour):
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        return len(approx) == 4 and cv.isContourConvex(approx), approx

    def detect_rectangles(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 50, 150)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for cnt in contours:
            is_rect, approx = self.is_rectangle(cnt)
            if is_rect:
                rectangles.append(approx)
        return rectangles


class HandTracker:
    def __init__(self, max_num_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=max_num_hands)
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


def point_inside_polygon(point, polygon):
    return cv.pointPolygonTest(polygon, point, False) >= 0


def main():
    cap = cv.VideoCapture(0)
    rect_detector = RectangleDetector()
    hand_tracker = HandTracker()

    rectangles = []
    rectangles_detected = False

    print("Press 'r' to detect rectangles in the current frame.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Detect rectangles once when user presses 'r'
        key = cv.waitKey(1) & 0xFF
        if key == ord('r'):
            rectangles = rect_detector.detect_rectangles(frame)
            rectangles_detected = True
            print(f"Detected {len(rectangles)} rectangle(s)")

        # Draw stored rectangles if detected
        if rectangles_detected:
            for rect in rectangles:
                cv.drawContours(frame, [rect], -1, (0, 255, 0), 3)

        # Process hand landmarks
        hand_results = hand_tracker.process(frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_tracker.draw_landmarks(frame, hand_landmarks)

                index_finger_tip = hand_landmarks.landmark[8]
                x_tip = int(index_finger_tip.x * w)
                y_tip = int(index_finger_tip.y * h)

                cv.circle(frame, (x_tip, y_tip), 8, (255, 0, 0), -1)

                # Check fingertip against stored rectangles
                if rectangles_detected:
                    for i, rect in enumerate(rectangles):
                        if point_inside_polygon((x_tip, y_tip), rect):
                            cv.putText(frame, f"Key {i+1} Pressed", (10, 40 + i*30),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv.drawContours(frame, [rect], -1, (0, 0, 255), 3)

        cv.imshow("Piano Key Detection", frame)

        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

