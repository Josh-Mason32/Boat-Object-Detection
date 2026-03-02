import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

# ==========================================
# CONFIGURATION & CALIBRATION PARAMETERS
# ==========================================
# Adjust these values to fine-tune detection sensitivity and performance

# 1. MODEL SETTINGS
CONFIDENCE_THRESHOLD = 0.15    # Stable baseline for balanced detection
IMAGE_SIZE = 640               # Standard high-performance resolution
VIDEO_STRIDE = 1               # Process every frame for maximum accuracy
IOU_THRESHOLD = 0.45           # Standard NMS overlap threshold

# 2. VIZ & ALERT SETTINGS
MAX_BOX_AREA_RATIO = 0.45      # Filter out boxes covering > 45% of frame (Amazing Filter)
MIN_FRAMES_FOR_ALERT = 2       # Persistence check for alert stability
MIN_CONF_FOR_ALERT = 0.15      # Confidence threshold for red banner
# Classes to ignore (names are matched case-insensitively)
EXCLUDE_CATEGORIES = ["static obstacle", "static objects", "sky", "water", "background"]

# 3. EXPORT SETTINGS
EXPORT_VIDEO = True            # Set to True to save the output to a file
# ==========================================

def predict_poc(source_video):
    print("--- YOLOv8 POC Predictor (Pi Deployment Version) ---")
    
    # --- Step 1: Initialize Paths ---
    script_path = Path(__file__).resolve()
    
    # Priority 1: Model weights in the same directory as the script
    model_weights = script_path.parent / "best.pt"
    
    if not model_weights.exists():
        print(f"ERROR: Could not find trained weights at {model_weights}.")
        print("Please ensure 'best.pt' is in the same folder as this script.")
        return

    # --- Step 2: Handle Source Video (Fallback to Pi Camera) ---
    if not source_video:
        # Default to the first camera (commonly the Pi Camera module or USB webcam on index 0)
        # Note: If it doesn't work, ensure libcamera is properly configured or use cv2.VideoCapture(0) instead of YOLO's stream
        print("No source video provided, falling back to default camera (index 0).")
        source_video = "0"
        global EXPORT_VIDEO
        EXPORT_VIDEO = False # Usually we don't save webcam feeds by default, but user can change this
    
    # If source is a file path, resolve it. If it's a digit (like "0"), treat it as a camera index.
    if str(source_video).isdigit():
        source_path = int(source_video) # Camera index
    else:
        source_path = Path(source_video).resolve()
        if not source_path.exists():
            print(f"ERROR: Source video not found: {source_video}")
            return

    # --- Step 3: Load Model ---
    print(f"Loading model: {model_weights}")
    model = YOLO(str(model_weights))
    
    print(f"Filtering: Area Ratio < {MAX_BOX_AREA_RATIO}, Exclude={EXCLUDE_CATEGORIES}")
    print(f"Sensitivity: Conf={CONFIDENCE_THRESHOLD}, Imgsz={IMAGE_SIZE}")

    # --- Step 4: Setup Export ---
    out = None
    if EXPORT_VIDEO and not isinstance(source_path, int):
        output_filename = f"amazing_output_{Path(source_path).stem}.mp4"
        cap_info = cv2.VideoCapture(str(source_path))
        width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_info.get(cv2.CAP_PROP_FPS)
        cap_info.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(script_path.parent / output_filename), fourcc, fps, (width, height))
        print(f"Saving output to: {output_filename}")
    else:
        # If camera, we will get dimensions on the fly if needed, but for simplicity we won't record by default
        pass

    # --- Step 5: Run Prediction Loop ---
    print("\nPress 'q' to quit.")
    
    source_to_pass = str(source_path) if not isinstance(source_path, int) else source_path
    
    # NOTE: We do NOT use the 'classes' filter here so the model sees everything
    results = model.track(
        source=source_to_pass, 
        stream=True, 
        conf=CONFIDENCE_THRESHOLD, 
        imgsz=IMAGE_SIZE, 
        iou=IOU_THRESHOLD,
        persist=True, 
        vid_stride=VIDEO_STRIDE,
        verbose=False
    )
    
    object_history = {}
    frame_area = None
    
    for r in results:
        # Create a clean copy for manual annotation
        annotated_frame = r.orig_img.copy()
        alert_triggered = False
        
        if frame_area is None:
            height, width = annotated_frame.shape[:2]
            frame_area = width * height
        
        if r.boxes:
            track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            confidences = r.boxes.conf.cpu().tolist()
            classes = r.boxes.cls.int().cpu().tolist()
            coords = r.boxes.xyxy.cpu().tolist()
            
            for box_coords, track_id, conf, cls in zip(coords, track_ids, confidences, classes):
                class_name = model.names[cls]
                
                # 1. Size-based suppression (The "Amazing" Filter)
                x1, y1, x2, y2 = map(int, box_coords)
                box_w, box_h = x2 - x1, y2 - y1
                box_area = box_w * box_h
                if box_area / frame_area > MAX_BOX_AREA_RATIO:
                    continue # Skip huge distracting boxes
                
                # 2. Category-based suppression
                if any(excl in class_name.lower() for excl in EXCLUDE_CATEGORIES):
                    continue
                
                # 3. Update alert logic
                if track_id is not None:
                    object_history[track_id] = object_history.get(track_id, 0) + 1
                    if conf >= MIN_CONF_FOR_ALERT and object_history[track_id] >= MIN_FRAMES_FOR_ALERT:
                        alert_triggered = True

                # 4. Manual Drawing (Restored for clean look)
                color = (0, 255, 0) # Professional Green
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {track_id if track_id else ''} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if alert_triggered:
            cv2.rectangle(annotated_frame, (0, 0), (350, 120), (0, 0, 0), -1)
            cv2.putText(annotated_frame, "ALERT!", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)
        
        if out:
            out.write(annotated_frame)
            
        cv2.imshow("YOLO Amazing Version Restoration (Pi)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"\nDone!")

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    predict_poc(video)
