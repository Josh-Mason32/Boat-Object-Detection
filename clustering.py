import cv2
import numpy as np
import argparse
import sys
import os

# -------------------------------
# CONFIG
# -------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 384

MIN_CONTOUR_AREA = 300  # Tune for distance (~100m)
MAX_CONTOUR_AREA = 50000

# HSV range for WATER (tune per environment)
LOWER_WATER = np.array([80, 20, 20])   # blue-green low sat
UPPER_WATER = np.array([140, 255, 255])

# -------------------------------
# ARGUMENT PARSING
# -------------------------------
parser = argparse.ArgumentParser(description="Object detection on water via clustering.")
parser.add_argument("source", nargs="?", default="0", help="Webcam index (e.g. 0) or path to video file.")
args = parser.parse_args()

# Try to convert source to int (for webcam); if it fails, treat as file path
try:
    source = int(args.source)
except ValueError:
    source = args.source
    if not os.path.exists(source):
        print(f"Error: Video file '{source}' not found.")
        sys.exit(1)

# -------------------------------
# VIDEO SETUP
# -------------------------------
cap = cv2.VideoCapture(source)

# If it's a file, we might not want to force these settings (some files have fixed sizes)
if isinstance(source, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=25,
    detectShadows=False
)

kernel = np.ones((5, 5), np.uint8)

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # 1️⃣ Motion detection
    motion_mask = fgbg.apply(frame)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    # 2️⃣ Water mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    water_mask = cv2.inRange(hsv, LOWER_WATER, UPPER_WATER)
    non_water_mask = cv2.bitwise_not(water_mask)

    # 3️⃣ Combine motion + non-water
    candidate_mask = cv2.bitwise_and(motion_mask, non_water_mask)

    # 4️⃣ Cleanup
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)

    # 5️⃣ Contour detection
    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect ratio filter (reject wave streaks)
        aspect_ratio = w / float(h)
        if aspect_ratio > 8 or aspect_ratio < 0.1:
            continue

        # Draw detection
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Object", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # -------------------------------
    # DISPLAY
    # -------------------------------
    cv2.imshow("Frame", frame)
    cv2.imshow("Motion Mask", motion_mask)
    cv2.imshow("Candidate Mask", candidate_mask)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

