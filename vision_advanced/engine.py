#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from utils import draw_landmarks_on_image



model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# --- Video Capture ---
feed = cv.VideoCapture(0)

while True:
    isTrue, frame = feed.read()
    if not isTrue:
        break

    # Convert BGR (OpenCV) to RGB (MediaPipe)
    flipped = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(flipped, cv.COLOR_BGR2RGB)

    # Wrap image for MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run hand detection
    detection_result = detector.detect(mp_image)

    # Draw landmarks
    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

    # Show annotated image (convert RGB back to BGR for OpenCV)
    cv.imshow("MediaPipe Hands", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

    if cv.waitKey(5) & 0xFF == 27:  # ESC key
        break

feed.release()
cv.destroyAllWindows()
