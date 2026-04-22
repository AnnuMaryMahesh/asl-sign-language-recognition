"""
collect_data.py
---------------
Run this FIRST to collect hand landmark data for each sign.

HOW TO USE:
1. Run: python collect_data.py
2. A webcam window will open
3. CLICK on the webcam window to give it focus
4. Make the ASL sign shown on screen
5. Press SPACEBAR (or the letter shown) to start collecting
6. Hold the sign still for ~3 seconds while it counts to 100
7. Repeat for each letter A-Z

Press Q anytime to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ── Config ──────────────────────────────────────────────────────────────────
SIGNS       = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER = 100
OUTPUT_FILE = "data/landmarks.csv"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs("data", exist_ok=True)

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.7)

# FIX 1: Use DirectShow backend for Windows camera compatibility
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    print("Try changing the index: cv2.VideoCapture(1, cv2.CAP_DSHOW) in this file.")
    exit()

# Write CSV header if file doesn't exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
        writer.writerow(header)

print("=== DATA COLLECTION ===")
print(f"Signs to collect: {SIGNS}")
print(f"Samples per sign: {SAMPLES_PER}")
print("\nINSTRUCTIONS:")
print("  1. When the webcam window opens, CLICK on it to give it focus")
print("  2. Make the hand sign shown on screen")
print("  3. Press SPACEBAR (or the letter key) to start collecting")
print("  4. Hold the sign still until the counter reaches 100\n")

for sign in SIGNS:
    collected = 0
    waiting   = True

    print(f"[{sign}]  Get ready... make the sign for '{sign}' then press SPACE or '{sign}'")

    while True:
        ret, frame = cap.read()

        # FIX 2: Exit cleanly if camera fails instead of silently skipping all signs
        if not ret:
            print("ERROR: Camera not accessible. Close other apps using the camera and try again.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # UI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if waiting:
            cv2.putText(frame, f"Sign: {sign}  |  Press SPACE or '{sign}' to start",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
        else:
            bar_w = int((collected / SAMPLES_PER) * (frame.shape[1] - 20))
            cv2.rectangle(frame, (10, 55), (10 + bar_w, 75), (0, 200, 80), -1)
            cv2.putText(frame, f"Collecting '{sign}': {collected}/{SAMPLES_PER}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Data Collection — Sign Language KNN  (Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("Quit by user.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # FIX 3: Accept lowercase letter, uppercase letter, OR spacebar to start
        if waiting and (key == ord(sign.lower()) or
                        key == ord(sign.upper()) or
                        key == 32):
            waiting = False
            time.sleep(0.3)

        if not waiting and results.multi_hand_landmarks:
            lm  = results.multi_hand_landmarks[0].landmark
            row = [coord for point in lm for coord in (point.x, point.y, point.z)]
            row.append(sign)
            with open(OUTPUT_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
            collected += 1

        if collected >= SAMPLES_PER:
            print(f"  ✓ Done — {SAMPLES_PER} samples saved for '{sign}'")
            break

cap.release()
cv2.destroyAllWindows()
print(f"\nAll data saved to: {OUTPUT_FILE}")
print("Next step: run train_model.py")
