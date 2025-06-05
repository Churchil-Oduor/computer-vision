import cv2
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe import solutions as mp
from mediapipe.framework.formats import landmark_pb2

# Drawing utils
mp_drawing = mp.drawing_utils
mp_styles = mp.drawing_styles

# Callback function for live stream mode
latest_result = None
def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Initialize the hand landmarker in live stream mode
BaseOptions = mp_tasks.base_options.BaseOptions
VisionRunningMode = mp_tasks.vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Send frame to the landmarker
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    hand_landmarker.detect_async(mp_image, timestamp_ms)

    # If we have results, draw them
    if latest_result and latest_result.hand_landmarks:
        for landmarks in latest_result.hand_landmarks:
            # Wrap list of NormalizedLandmark into NormalizedLandmarkList
            landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_list,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_styles.get_default_hand_connections_style()
            )

    # Show the image
    cv2.imshow("Hand Landmarker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

