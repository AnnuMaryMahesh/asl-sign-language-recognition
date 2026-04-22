"""
recognize.py
------------
Run this THIRD for real-time sign language recognition.

Opens your webcam, detects hand landmarks via MediaPipe, and classifies
the sign using the trained KNN model.

Controls:
  Q          — Quit
  B / BKSP   — Delete last letter
  C          — Clear word
  SPACE      — Add a space
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import collections
import time

MODEL_FILE  = "data/knn_model.pkl"
SCALER_FILE = "data/scaler.pkl"

# ── Load Model ────────────────────────────────────────────────────────────────
with open(MODEL_FILE,  "rb") as f: knn    = pickle.load(f)
with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)

print("Model loaded. Opening webcam...")
print("Controls: Q = quit | B = backspace | C = clear | SPACE = space\n")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.75,
                             min_tracking_confidence=0.75)

# ── Smoothing: majority vote over last 10 frames ──────────────────────────────
SMOOTH_N   = 10
pred_queue = collections.deque(maxlen=SMOOTH_N)

# ── Word builder ──────────────────────────────────────────────────────────────
word        = ""
last_letter = ""
letter_hold = 0
HOLD_FRAMES = 20

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = (0, 210,  80)
CYAN   = (0, 230, 255)
WHITE  = (255, 255, 255)
DARK   = (20,  20,  20)
ORANGE = (0, 165, 255)

# FIX: Use DirectShow backend for Windows compatibility
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

fps_time    = time.time()
frame_count = 0
fps         = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_sign = None
    confidence     = 0.0

    if result.multi_hand_landmarks:
        hl = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hl, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=WHITE, thickness=2)
        )

        lm  = hl.landmark
        row = [coord for point in lm for coord in (point.x, point.y, point.z)]
        row_scaled = scaler.transform([row])

        predicted_sign = knn.predict(row_scaled)[0]
        proba          = knn.predict_proba(row_scaled)[0]
        confidence     = max(proba)

        pred_queue.append(predicted_sign)

        if len(pred_queue) == SMOOTH_N:
            smoothed = collections.Counter(pred_queue).most_common(1)[0][0]
        else:
            smoothed = predicted_sign

        if smoothed == last_letter:
            letter_hold += 1
        else:
            letter_hold = 0
            last_letter = smoothed

        if letter_hold == HOLD_FRAMES:
            word += smoothed

        predicted_sign = smoothed

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 90), DARK, -1)

    sign_text = predicted_sign if predicted_sign else "—"
    cv2.putText(frame, sign_text, (20, 72),
                cv2.FONT_HERSHEY_DUPLEX, 2.8, CYAN, 4)

    if predicted_sign:
        bar_x, bar_y, bar_h = 110, 20, 20
        bar_max   = 300
        bar_fill  = int(confidence * bar_max)
        bar_color = GREEN if confidence > 0.7 else ORANGE
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"{confidence*100:.0f}%",
                    (bar_x + bar_max + 8, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.putText(frame, "Confidence",
                    (bar_x, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.rectangle(frame, (0, h - 70), (w, h), DARK, -1)
    cv2.putText(frame, "Word: " + word, (15, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
    cv2.putText(frame, "B = backspace   C = clear   SPACE = space   Q = quit",
                (15, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    frame_count += 1
    if time.time() - fps_time >= 1.0:
        fps         = frame_count
        frame_count = 0
        fps_time    = time.time()
    cv2.putText(frame, f"FPS: {fps}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1)

    cv2.imshow("Real-Time Sign Language Recognition  |  Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('b') or key == 8:
        word = word[:-1]
    elif key == ord('c') or key == ord('C'):
        word = ""
    elif key == 32:
        word += " "

cap.release()
cv2.destroyAllWindows()
print("Exited.")
