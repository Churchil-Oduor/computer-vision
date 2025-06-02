import cv2 as cv
from utils import RectangleDetector
from utils import HandTracker
from utils import point_inside_rectangle
#from playsound import playsound
import threading
from utils import play_note

def main():
    feed = cv.VideoCapture(0)
    rect_detector = RectangleDetector()
    hand_tracker = HandTracker()

    rectangles = []
    rectangles_Detected = False

    while True: 
        isTrue, frame = feed.read()
        if not isTrue:
            break

        frame = cv.flip(frame, 1)

        h, w, _ = frame.shape
        key = cv.waitKey(1) & 0xff

        # detecting rectangles when the r key is pressed
        if key == ord('r'):
            rectangles = rect_detector.detect_rectangles(frame)
            rectangles_Detected = True
            print(f"Detected {len(rectangles)} rectangle(s)")


        # drawing stored rectangles if detected
        if rectangles_Detected:
            for rect in rectangles:
                cv.drawContours(frame, [rect], -1, (0, 255, 0), 3)

        hand_results = hand_tracker.process(frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_tracker.draw_landmarks(frame, hand_landmarks)

                index_finger_tip = hand_landmarks.landmark[8]
                x_tip = int(index_finger_tip.x * w)
                y_tip = int(index_finger_tip.y * h)

                cv.circle(frame, (x_tip, y_tip), 8, (255, 0, 0), -1)

                if rectangles_Detected:
                    for i, rect in enumerate(rectangles):
                        if point_inside_rectangle((x_tip, y_tip), rect):
                            cv.putText(frame, f"Key {i+1} pressed", (10, 40 + i* 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2),
                            cv.drawContours(frame, [rect], -1, (0,0, 255), 3)
                            threading.Thread(target=play_note, daemon=True).start()

        cv.imshow("Piano key detection", frame)

        if key == ord('q'):
            break

    feed.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
