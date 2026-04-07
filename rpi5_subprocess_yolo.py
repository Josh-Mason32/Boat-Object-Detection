import cv2
import sys
import time
import subprocess
import numpy as np
from ultralytics import YOLO
from pathlib import Path

OUTPUT_PATH = "output_recording.mp4"
ALERT_SOUND_PATH = "alert.wav"

MIN_FRAMES_FOR_ALERT = 3
MIN_CONF_FOR_ALERT   = 0.15
EXCLUDE_CATEGORIES   = ["static obstacle", "static objects", "sky", "water", "background"]
MAX_BOX_AREA_RATIO   = 0.45
MIDDLE_COLUMN_ONLY   = True
MIDDLE_COLUMN_RATIO  = 0.9
ALERT_DURATION       = 5
HEADLESS_MODE        = False # Set to True to run automatically on boot with no monitor needed

def play_alert_sound():
    try:
        subprocess.Popen(["aplay", "-q", ALERT_SOUND_PATH])
    except:
        pass

def main():
    print("[*] Starting Hardware Capture Subprocess...")
    
    # We use rpicam-vid to completely bypass Python camera bugs.
    # It runs natively on the Pi and streams raw video data to stdout.
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--inline",
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--codec", "yuv420",
        "-o", "-"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    # Raw YUV420 frame size calculation
    # Width x Height x 1.5
    # 640 * 480 * 1.5 = 460800 bytes
    FRAME_BYTES = 460800
    
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

    # Play two startup beeps in the background
    try:
        subprocess.Popen(["sh", "-c", f"aplay -q '{ALERT_SOUND_PATH}'; sleep 0.4; aplay -q '{ALERT_SOUND_PATH}'"])
    except:
        pass

    object_history = {}
    frame_area = None
    smoothed_horizon_y = None
    alert_until_time = 0

    while True:
        try:
            # Read exact frame bytes from the camera subprocess
            raw_data = process.stdout.read(FRAME_BYTES)
            
            if len(raw_data) != FRAME_BYTES:
                print("\n[*] Camera feed ended or disconnected.")
                break
                
            # Convert raw bytes to standard OpenCV BGR image
            yuv_frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((720, 640))
            frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

            # --- exact same tracking logic ---
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

            if not HEADLESS_MODE:
                cv2.imshow("Video Prediction", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            break

    print("[*] Releasing resources and saving recording.")
    process.terminate()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
