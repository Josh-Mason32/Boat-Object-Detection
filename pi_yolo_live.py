import cv2
import sys
import threading
import time
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# RASPBERRY PI CAMERA CONFIGURATION
# ==========================================
# 0 is usually the default camera (CSI or USB).
# If you have multiple cameras, try 1, 2, etc.
# On newer Pi OS (Bookworm), you might need to use:
# STREAM_URL = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! appsink"
STREAM_URL = 0 
# ==========================================

# ==========================================
# YOLO ALERT LOGIC SETTINGS (Improved Version)
# ==========================================
MIN_FRAMES_FOR_ALERT = 2       # How many consecutive frames before alert triggers
MIN_CONF_FOR_ALERT   = 0.15   # Minimum confidence to count toward alert
EXCLUDE_CATEGORIES   = ["static obstacle", "static objects", "sky", "water", "background"]
MAX_BOX_AREA_RATIO   = 0.45   # Ignore boxes covering > 45% of frame
MIDDLE_COLUMN_ONLY   = True   # Only alert for objects in the middle column
MIDDLE_COLUMN_RATIO  = 0.40   # Width of the middle column (e.g., 0.40 = middle 40% of screen)

# PERFORMANCE OPTIMIZATION
IMGSZ = 320 # Smaller size for better CPU performance on Pi
# ==========================================

class VideoStreamWidget(object):
    """
    A threaded video capture widget. This reads frames constantly in the background
    so the camera buffer never fills up and causes lag on the Pi.
    """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        if not self.capture.isOpened():
            print(f"\n[X] Failed to open camera at source: {src}")
            print("[TIP] If on Raspberry Pi Bullseye/Bookworm, ensure libcamera-apps is installed.")
            print("[TIP] You may need to run: libcamerify python pi_yolo_live.py")
            sys.exit(1)
            
        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(0.01)

def main():
    print(f"[*] Starting Pi Camera Feed (Source: {STREAM_URL})")
    video_stream = VideoStreamWidget(STREAM_URL)
    
    script_dir = Path(__file__).resolve().parent
    # Prefer best.pt if available, fallback to og.pt
    model_path = script_dir / "best.pt"
    if not model_path.exists():
        model_path = script_dir / "og.pt"
    
    print(f"[*] Loading model from: {model_path}")
    model = YOLO(str(model_path))

    print("[+] System Ready! Displaying live prediction feed.")
    print("[▶] Press 'q' in the window to quit.\n")

    object_history = {}
    frame_area = None
    smoothed_horizon_y = None

    while True:
        try:
            # Grab latest frame from the threaded fetcher
            frame = video_stream.frame
            if frame is None:
                continue
                
            # Need a copy to annotate on
            process_frame = frame.copy()
            
            # Run YOLO tracking
            results = model.track(
                source=process_frame,
                persist=True,
                verbose=False,
                imgsz=IMGSZ
            )

            # --- YOLO Alert Logic ---
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

                    # Update smoothed horizon 
                    if current_horizon_candidates:
                        frame_horizon = sum(current_horizon_candidates) / len(current_horizon_candidates)
                        if smoothed_horizon_y is None:
                            smoothed_horizon_y = frame_horizon
                        else:
                            # EMA smoothing to prevent jitter
                            smoothed_horizon_y = 0.9 * smoothed_horizon_y + 0.1 * frame_horizon

                # Draw the final overlay
                if alert_triggered:
                    cv2.rectangle(annotated, (0, 0), (350, 120), (0, 0, 0), -1)
                    cv2.putText(annotated, "Dats an object", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)

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
                
                # Status text
                cv2.putText(
                    annotated, "Pi Live Feed",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA
                )

                # Show frame
                cv2.imshow("Raspberry Pi YOLO Live", annotated)

            # Wait for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_stream.capture.release()
                cv2.destroyAllWindows()
                break

        except AttributeError:
            # Reached before the first frame is grabbed
            time.sleep(0.1)
            pass
        except KeyboardInterrupt:
            video_stream.capture.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
