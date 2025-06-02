import mediapipe as mp
import cv2 as cv
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

feed = cv.VideoCapture(0)

while True:
    frame_check, frame = feed.read()
    
    if not frame_check:
        break

    flipped_frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(flipped_frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv.imshow("hand tracking", flipped_frame)

    if cv.waitKey(1) & 0xff == ord('d'):
        break
feed.release()
cv.destroyAllWindows()
