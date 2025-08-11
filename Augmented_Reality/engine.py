#!/usr/bin/env python3
import cv2 as cv
import mediapipe as mp

class HandTrack:
    def __init__(self, mode=False, maxHands=2, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        self.mode = mode
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.maxHands = maxHands
        self.hands = self.mpHands.Hands(mode, maxHands, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def handGesture(self, frame, draw=True):
        """
        Detects the hand(s)

        frame: captured Frame image
        handNo: index of hand to be displayed.
        draw: whether to draw landmark over hand or not.

        Return: Returns the landmarks.

        """
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        lmList = []
        fingerTips = [4, 8, 12, 16, 20]
        left_hand = None
        right_hand = None
        self.my_hands = {
                "left_hand": [],
                "right_hand": []
                }

        if results.multi_hand_landmarks:
            for idx, handlms in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                handKey = "left_hand"
                if handedness == "Left":
                    left_hand = handlms
                    handKey = "left_hand"
                elif handedness == "Right":
                    right_hand = handlms
                    handKey = "right_hand"

                
                for tip_id in fingerTips:
                    h, w, _ = frame.shape
                    lm = handlms.landmark[tip_id]
                    x, y = int(lm.x * w), int(lm.y * h)
                    self.my_hands[handKey].append([tip_id, x, y])


                #color left hand blue for distinction
                if left_hand:
                    self.mpDraw.draw_landmarks(frame, left_hand, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                
                # color right hand to green for distintion
                if right_hand:
                    self.mpDraw.draw_landmarks(frame, right_hand, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(255, 255, 6), thickness=2, circle_radius=2))    

                #detected_hands = self.handStatus(frame, idx, left_hand, right_hand)
                print(self.my_hands)
        return frame, self.my_hands

    def handStatus(self, frame, idx, left_hand=None, right_hand=None):
        """
        checks three hand statuses, hold, rotate, x,y axis and rotate z axix.
        left hand controls z, axis and right hand rotation about the xy axis.
        """
        self.my_hands = {
                "left_hand": {"co-ord": [],
                              "tips": {}
                              },
                "right_hand": {"co-ord": [],
                               "tips": {}
                               }
                }

        h, w, _ = frame.shape
        fingerTips = [4, 8, 12, 16, 20]
        if left_hand:
            #left hand
            lft_x, lft_y = int(left_hand.landmark[0].x * w), int(left_hand.landmark[0].y * h)
            self.my_hands["left_hand"].append([idx, lft_x, lft_y])
#            print(self.my_hands["left_hand"])

        if right_hand:
            # right hand
            rt_x, rt_y = int(right_hand.landmark[0].x * w), int(right_hand.landmark[0].y * h)
            self.my_hands["right_hand"].append([idx, rt_x, rt_y])

        return self.my_hands

