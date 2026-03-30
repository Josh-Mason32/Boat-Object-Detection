import cv2
import sys
import threading
import time
import subprocess
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# VIDEO INPUT / OUTPUT CONFIGURATION
# ==========================================
# Provide the camera index for Pi Camera HQ (usually 0 if V4L2 is enabled)
# Or a GStreamer pipeline string if using libcamera natively through OpenCV
VIDEO_SOURCE = 0
# The path where the recorded predictions will be saved
OUTPUT_PATH = "output_recording.mp4"
# The path to the alert sound to play (.wav format is best for natively playing on rpi)
ALERT_SOUND_PATH = "alert.wav"
# ==========================================

# ==========================================
# YOLO ALERT LOGIC SETTINGS
# ==========================================
MIN_FRAMES_FOR_ALERT = 3       # How many consecutive frames before alert triggers
MIN_CONF_FOR_ALERT   = 0.15   # Minimum confidence to count toward alert
EXCLUDE_CATEGORIES   = ["static obstacle", "static objects", "sky", "water", "background"]
MAX_BOX_AREA_RATIO   = 0.45   # Ignore boxes covering > 45% of frame
MIDDLE_COLUMN_ONLY   = True   # Only alert for objects in the middle column
MIDDLE_COLUMN_RATIO  = 0.20   # Width of the middle column (e.g., 0.40 = middle 40% of screen)
ALERT_DURATION       = 5      # How long the alert stays on screen (seconds)
# ==========================================

def play_alert_sound():
    try:
        # aplay is the default ALSA audio player on Raspberry Pi
        subprocess.Popen(["aplay", "-q", ALERT_SOUND_PATH])
    except Exception as e:
        print(f"[!] Failed to play audio: {e}")

def main():
    print(f"[*] Opening camera source: {VIDEO_SOURCE}")
    
    capture = cv2.VideoCapture(VIDEO_SOURCE)
    # Give the camera a moment to warm up
    time.sleep(2.0)
    
    if not capture.isOpened():
        print(f"\n[X] Failed to open camera source: {VIDEO_SOURCE}")
        sys.exit(1)

    # Get video properties for output writer
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    # If fps is missing or invalid, default to 30
    if not fps or fps <= 0:
        fps = 30.0

    print(f"[*] Video Properties: {frame_width}x{frame_height} at {fps} FPS")
    print(f"[*] Initializing recording to: {OUTPUT_PATH}")
    
    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "og.pt"
    
    print(f"[*] Loading original YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    print("[+] System Ready! Displaying video prediction feed and recording.")
    print("[▶] Press 'q' in the window to quit.\n")

    object_history = {}
    frame_area = None
    smoothed_horizon_y = None
    alert_until_time = 0

    while True:
        try:
            # Read the next frame from the video
            ret, frame = capture.read()
            
            # If the video has finished or camera disconnected
            if not ret or frame is None:
                print("\n[*] Camera feed ended or disconnected.")
                break
                
            # Run YOLO tracking on this single frame
            results = model.track(
                source=frame,
                persist=True,
                verbose=False
            )

            # Define a variable to hold the annotated frame
            annotated = frame.copy()

            current_time = time.time()
            
            # If a previous alert just finished, reset tracking for a fresh search
            if alert_until_time > 0 and current_time >= alert_until_time:
                alert_until_time = 0
                object_history.clear()

            is_in_alert_window = (current_time < alert_until_time)

            # --- YOLO Alert Logic (from og.py) ---
            for r in results:
                annotated = r.plot()
                alert_triggered = False

                if frame_area is None and r.orig_img is not None:
                    h, w = r.orig_img.shape[:2]
                    frame_area = w * h

                if r.boxes and frame_area:
                    # Parse bounding box tracking info
                    track_ids   = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
                    confidences = r.boxes.conf.cpu().tolist()
                    classes     = r.boxes.cls.int().cpu().tolist()
                    coords      = r.boxes.xyxy.cpu().tolist()

                    current_horizon_candidates = []

                    for box_coords, track_id, conf, cls in zip(coords, track_ids, confidences, classes):
                        class_name = model.names[cls]

                        # Size filter
                        x1, y1, x2, y2 = map(int, box_coords)
                        box_area = (x2 - x1) * (y2 - y1)

                        # Gather water/sky candidates for horizon tracking
                        if "water" in class_name.lower():
                            current_horizon_candidates.append(y1)
                        if "sky" in class_name.lower():
                            current_horizon_candidates.append(y2)

                        if (box_area / frame_area) > MAX_BOX_AREA_RATIO:
                            continue

                        # Middle column filter
                        if MIDDLE_COLUMN_ONLY:
                            box_center_x = (x1 + x2) / 2
                            frame_w = r.orig_img.shape[1]
                            edge_margin = (1.0 - MIDDLE_COLUMN_RATIO) / 2.0
                            left_bound = frame_w * edge_margin
                            right_bound = frame_w * (1.0 - edge_margin)
                            if not (left_bound <= box_center_x <= right_bound):
                                continue

                        # Category filter
                        if any(excl in class_name.lower() for excl in EXCLUDE_CATEGORIES):
                            continue

                        # Persistence tracking
                        if track_id is not None:
                            object_history[track_id] = object_history.get(track_id, 0) + 1
                            if conf >= MIN_CONF_FOR_ALERT and object_history[track_id] >= MIN_FRAMES_FOR_ALERT:
                                alert_triggered = True

                    # Trigger the timed alert window if not already in one
                    if alert_triggered and not is_in_alert_window:
                        alert_until_time = current_time + ALERT_DURATION
                        is_in_alert_window = True
                        play_alert_sound()

                    # Update smoothed horizon 
                    if current_horizon_candidates:
                        frame_horizon = sum(current_horizon_candidates) / len(current_horizon_candidates)
                        if smoothed_horizon_y is None:
                            smoothed_horizon_y = frame_horizon
                        else:
                            # EMA smoothing to prevent jitter
                            smoothed_horizon_y = 0.9 * smoothed_horizon_y + 0.1 * frame_horizon

                # Draw the final overlay
                if is_in_alert_window:
                    cv2.rectangle(annotated, (0, 0), (350, 120), (0, 0, 0), -1)
                    cv2.putText(annotated, "Object", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)
                
                cv2.putText(
                    annotated, "Analyzing Video",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA
                )

                # Draw middle column guidelines
                if MIDDLE_COLUMN_ONLY and r.orig_img is not None:
                    h_img, w_img = r.orig_img.shape[:2]
                    edge_margin = (1.0 - MIDDLE_COLUMN_RATIO) / 2.0
                    left_bound = int(w_img * edge_margin)
                    right_bound = int(w_img * (1.0 - edge_margin))
                    cv2.line(annotated, (left_bound, 0), (left_bound, h_img), (255, 255, 0), 2)
                    cv2.line(annotated, (right_bound, 0), (right_bound, h_img), (255, 255, 0), 2)

                # Draw dynamic water horizon
                if smoothed_horizon_y is not None and r.orig_img is not None:
                    hy = int(smoothed_horizon_y)
                    w_img = r.orig_img.shape[1]
                    cv2.line(annotated, (0, hy), (w_img, hy), (255, 100, 0), 2)
                    cv2.putText(annotated, "Water Horizon", (10, hy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA)

            # Write the frame into the output video file
            out_writer.write(annotated)

            # Show frame
            cv2.imshow("Video Prediction", annotated)

            # Wait for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

    # Clean up resources
    print("[*] Releasing resources and saving recording.")
    capture.release()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
