import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

eastern_egg_image = cv2.imread('sixseven.png')

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
    frame = cv2.resize(frame, (1000, 720))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    finger_counts = []  

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            finger_count = 0
            lm = hand_landmarks.landmark

            if lm[4].x > lm[3].x:
                finger_count += 1

            for tip in finger_tips[1:]:
                if lm[tip].y < lm[tip - 2].y:
                    finger_count += 1

            finger_counts.append(finger_count)
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

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
    is_easter_egg = False

    if len(values) == 0:
        display_text = "< > + < >"
    elif len(values) == 1:
        display_text = f"{values[0]} +"
    else:
        result = values[0] + values[1]
        display_text = f"{values[0]} + {values[1]} = {result}"

        if values[0] == 6 and values[1] == 7:
            is_easter_egg = True

    cv2.putText(
        frame,
        display_text,
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    if is_easter_egg and eastern_egg_image is not None:
        img_h, img_w = eastern_egg_image.shape[:2]
        max_img_w, max_img_h = 300, 300
        scale = min(max_img_w / img_w, max_img_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = cv2.resize(eastern_egg_image, (new_w, new_h))
        x_offset = (w - new_w) // 2
        y_offset = h - new_h - 20
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    cv2.putText(
        frame,
        f"Hand 1: {finger_counts[0] if len(finger_counts) > 0 else 0} | Hand 2: {finger_counts[1] if len(finger_counts) > 1 else 0}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    total_fingers = sum(finger_counts)

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
        "Press 'Y' to Reset",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Finger Algebra Input", frame)

    key = cv2.waitKey(1) & 0xFF

    if total_fingers > 0 and len(values) < 2 and capture_start_time is None:
        capture_start_time = time.time()

    if capture_start_time is not None and len(values) < 2:
        elapsed = time.time() - capture_start_time
        if elapsed >= capture_delay:
            values.append(total_fingers)
            capture_start_time = None

    if total_fingers == 0:
        capture_start_time = None

    if key == ord('y') or key == ord('Y'):
        values = []
        capture_start_time = None

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
