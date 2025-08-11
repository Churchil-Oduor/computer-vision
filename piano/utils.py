import cv2 as cv
import mediapipe as mp
#from playsound import playsound


class RectangleDetector:
    """
    Detects the drawn keys

    """

    def __init__(self):
        pass

    def is_rectangle(self, contour):
        """
        checks to see if the figure is a rectangle

        contour: received contour from frame.
        Returns: a tuple containing a bool and approximation of the 
        rectangle
        """
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
            frame_check, approx = self.is_rectangle(cnt)
            if frame_check:
                rectangles.append(approx)
        return rectangles

def point_inside_rectangle(point, polygon):
    return cv.pointPolygonTest(polygon, point, False) >= 0


class HandTracker:
    def __init__(self, max_num_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


def play_note():
    playsound("note.mp3")

