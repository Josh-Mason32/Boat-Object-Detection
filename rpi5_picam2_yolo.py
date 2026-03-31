import cv2
import sys
import time
import subprocess
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Try to import the Official Raspberry Pi 5 Camera Library
try:
    from picamera2 import Picamera2
except ImportError:
    print("[X] The 'picamera2' library is missing.")
    print("    Please install it using: sudo apt install python3-picamera2")
    sys.exit(1)

OUTPUT_PATH = "output_recording.mp4"
ALERT_SOUND_PATH = "alert.wav"

MIN_FRAMES_FOR_ALERT = 3
MIN_CONF_FOR_ALERT   = 0.15
EXCLUDE_CATEGORIES   = ["static obstacle", "static objects", "sky", "water", "background"]
MAX_BOX_AREA_RATIO   = 0.45
MIDDLE_COLUMN_ONLY   = True
MIDDLE_COLUMN_RATIO  = 0.20
ALERT_DURATION       = 5

def play_alert_sound():
    try:
        subprocess.Popen(["aplay", "-q", ALERT_SOUND_PATH])
    except:
        pass

def main():
    print("[*] Initializing Native Raspberry Pi 5 Camera (Picamera2)...")
    
    # Setup Picamera2 engine
    picam2 = Picamera2()
    # Configure it to request a simple 640x480 RGB stream
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    # We will let picam warmup
    time.sleep(2.0)
    
    frame_width = 640
    frame_height = 480
    fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "og.pt"
    
    print(f"[*] Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    print("[+] System Ready! Displaying video prediction feed and recording.")
    print("[▶] Press 'q' in the window to quit.\n")

    object_history = {}
    frame_area = None
    smoothed_horizon_y = None
    alert_until_time = 0

    while True:
        try:
            # Native capture request from Pi 5 hardware
            try:
                frame_rgb = picam2.capture_array()
            except RuntimeError as e:
                print(f"\n[!] Camera feed disconnected or error: {e}")
                break
                
            # Convert the Pi's RGB array into OpenCV's standard BGR array
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Run YOLO tracking on this single frame
            results = model.track(
                source=frame,
                persist=True,
                verbose=False
            )

            annotated = frame.copy()
            current_time = time.time()
            
            if alert_until_time > 0 and current_time >= alert_until_time:
                alert_until_time = 0
                object_history.clear()

            is_in_alert_window = (current_time < alert_until_time)

            for r in results:
                annotated = r.plot()
                alert_triggered = False

                if frame_area is None and r.orig_img is not None:
                    h, w = r.orig_img.shape[:2]
                    frame_area = w * h

                if r.boxes and frame_area:
                    track_ids   = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
                    confidences = r.boxes.conf.cpu().tolist()
                    classes     = r.boxes.cls.int().cpu().tolist()
                    coords      = r.boxes.xyxy.cpu().tolist()

                    current_horizon_candidates = []

                    for box_coords, track_id, conf, cls in zip(coords, track_ids, confidences, classes):
                        class_name = model.names[cls]

                        x1, y1, x2, y2 = map(int, box_coords)
                        box_area = (x2 - x1) * (y2 - y1)

                        if "water" in class_name.lower():
                            current_horizon_candidates.append(y1)
                        if "sky" in class_name.lower():
                            current_horizon_candidates.append(y2)

                        if (box_area / frame_area) > MAX_BOX_AREA_RATIO:
                            continue

                        if MIDDLE_COLUMN_ONLY:
                            box_center_x = (x1 + x2) / 2
                            frame_w = r.orig_img.shape[1]
                            edge_margin = (1.0 - MIDDLE_COLUMN_RATIO) / 2.0
                            left_bound = frame_w * edge_margin
                            right_bound = frame_w * (1.0 - edge_margin)
                            if not (left_bound <= box_center_x <= right_bound):
                                continue

                        if any(excl in class_name.lower() for excl in EXCLUDE_CATEGORIES):
                            continue

                        if track_id is not None:
                            object_history[track_id] = object_history.get(track_id, 0) + 1
                            if conf >= MIN_CONF_FOR_ALERT and object_history[track_id] >= MIN_FRAMES_FOR_ALERT:
                                alert_triggered = True

                    if alert_triggered and not is_in_alert_window:
                        alert_until_time = current_time + ALERT_DURATION
                        is_in_alert_window = True
                        play_alert_sound()

                    if current_horizon_candidates:
                        frame_horizon = sum(current_horizon_candidates) / len(current_horizon_candidates)
                        if smoothed_horizon_y is None:
                            smoothed_horizon_y = frame_horizon
                        else:
                            smoothed_horizon_y = 0.9 * smoothed_horizon_y + 0.1 * frame_horizon

                if is_in_alert_window:
                    cv2.rectangle(annotated, (0, 0), (350, 120), (0, 0, 0), -1)
                    cv2.putText(annotated, "Object", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)
                
                cv2.putText(
                    annotated, "Analyzing Video",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA
                )

                if MIDDLE_COLUMN_ONLY and r.orig_img is not None:
                    h_img, w_img = r.orig_img.shape[:2]
                    edge_margin = (1.0 - MIDDLE_COLUMN_RATIO) / 2.0
                    left_bound = int(w_img * edge_margin)
                    right_bound = int(w_img * (1.0 - edge_margin))
                    cv2.line(annotated, (left_bound, 0), (left_bound, h_img), (255, 255, 0), 2)
                    cv2.line(annotated, (right_bound, 0), (right_bound, h_img), (255, 255, 0), 2)

                if smoothed_horizon_y is not None and r.orig_img is not None:
                    hy = int(smoothed_horizon_y)
                    w_img = r.orig_img.shape[1]
                    cv2.line(annotated, (0, hy), (w_img, hy), (255, 100, 0), 2)
                    cv2.putText(annotated, "Water Horizon", (10, hy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA)

            out_writer.write(annotated)

            cv2.imshow("Video Prediction", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

    print("[*] Releasing resources and saving recording.")
    picam2.stop()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
