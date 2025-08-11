import numpy as np
import cv2 as cv
import mediapipe as mp
import utils as utl

class HandTrack:
    
    def __init__(self, mode=False, maxHands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = mode
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.maxHands = maxHands
        self.hands = self.mpHands.Hands(
            mode, maxHands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.base_line = [(10, 0), (20, 0)]
        self.calc = utl.Calculations()

    def handPositions(self, frame, draw=True):
        """
        Detects left/right hands and fingertip positions.

        Returns:
            frame: annotated frame
            self.my_hands: dict with fingertip coordinates
        """
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        finger_tips_ids = np.array([4, 8, 12, 16, 20])
        h, w, _ = frame.shape

        self.my_hands = {
            "left_hand": np.empty((0, 3), dtype=int),   # tip_id, x, y
            "right_hand": np.empty((0, 3), dtype=int)
        }

        if results.multi_hand_landmarks:
            for idx, handlms in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                hand_key = "left_hand" if handedness == "Left" else "right_hand"

                # Convert all landmarks to numpy array (21, 2)
                coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in handlms.landmark], dtype=int)

                # Extract fingertip coordinates and store as [tip_id, x, y]
                tips_array = np.column_stack((finger_tips_ids, coords[finger_tips_ids]))
                self.my_hands[hand_key] = tips_array

                # Draw if enabled
                if draw:
                    color = (255, 0, 0) if hand_key == "left_hand" else (0, 255, 0)
                    self.mpDraw.draw_landmarks(
                        frame, handlms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=color, thickness=2, circle_radius=2)
                    )
                self.handStatus(frame)

        return frame, self.my_hands

    def handStatus(self, frame):
        """
        Detects the fingertips to determine the hand status
        
        Returns:
            
        """
        right_hand = self.my_hands["right_hand"]
        left_hand = self.my_hands["left_hand"]

        if len(right_hand) > 0:
            right_index = right_hand[1]
            #right_thumb = right_hand[0]

        if len(left_hand) > 0:
            left_index = left_hand[1]
            #left_thumb = left_hand[0]

        if len(left_hand) > 0 and len(right_hand) > 0:
            x1, y1 = left_index[1], left_index[2]   # Left index tip
            #x2, y2 = left_thumb[1], left_thumb[2]   # Left thumb tip
            x3, y3 = right_index[1], right_index[2] # Right index tip
            #x4, y4 = right_thumb[1], right_thumb[2] # Right thumb tip
            line_1 = [(x1, y1), (x3, y3)]
            angle = self.calc.rotationAngle(self.base_line, line_1)
            cv.line(frame, (x1, y1), (x3, y3), (255, 7, 6), 3)

