import cv2
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions as mp  # For drawing utils

mp_drawing = mp.drawing_utils
mp_styles = mp.drawing_styles

latest_result = None

def print_result(result: vision.HandLandmarkerResult, output_image, timestamp_ms: int):
    global latest_result
    latest_result = result

BaseOptions = mp_tasks.BaseOptions
VisionRunningMode = mp_tasks.vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

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

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Pass numpy ndarray directly — no vision.Image
#    hand_landmarker.detect_async(rgb_frame, timestamp_ms, )

    if latest_result and latest_result.hand_landmarks:
        for hand_landmarks_proto in latest_result.hand_landmarks:
            landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=hand_landmarks_proto)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_list,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_styles.get_default_hand_connections_style()
            )

    cv2.imshow("Hand Landmarker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

