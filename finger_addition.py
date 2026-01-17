import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]

values = []
capture_start_time = None
capture_delay = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 720))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    finger_count = 0
    is_peace_pose = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            if lm[4].x > lm[3].x:
                finger_count += 1

            for tip in finger_tips[1:]:
                if lm[tip].y < lm[tip - 2].y:
                    finger_count += 1

            if (lm[8].y < lm[6].y and lm[12].y < lm[10].y and 
                lm[16].y > lm[14].y and lm[20].y > lm[18].y and 
                lm[4].x < lm[3].x):
                is_peace_pose = True

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # ---------------- UI TEXT ---------------- #

    cv2.putText(
        frame,
        "Please enter a number using your fingers",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    h, w, _ = frame.shape
    display_text = ""

    if len(values) == 0:
        display_text = "< > + < >"
    elif len(values) == 1:
        display_text = f"{values[0]} +"
    else:
        result = values[0] + values[1]
        display_text = f"{values[0]} + {values[1]} = {result}"

    cv2.putText(
        frame,
        display_text,
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Detected: {finger_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if capture_start_time is not None:
        elapsed = time.time() - capture_start_time
        remaining = max(0, capture_delay - elapsed)
        countdown = int(remaining) + 1
        
        cv2.putText(
            frame,
            f"Capturing in: {countdown}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "Peace Pose (✌️) to Reset",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Finger Algebra Input", frame)

    key = cv2.waitKey(1) & 0xFF

    if finger_count > 0 and len(values) < 2 and capture_start_time is None:
        capture_start_time = time.time()

    if capture_start_time is not None and len(values) < 2:
        elapsed = time.time() - capture_start_time
        if elapsed >= capture_delay:
            values.append(finger_count)
            capture_start_time = None

    if finger_count == 0:
        capture_start_time = None

    if is_peace_pose:
        values = []
        capture_start_time = None

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
