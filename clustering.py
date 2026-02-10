import cv2
import numpy as np
import argparse
import sys
import os
import time

# -----------------------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------------------
# Dimensions to resize the video to. Smaller = Faster processing.
FRAME_WIDTH = 640
FRAME_HEIGHT = 384

# Minimum area (pixels) for a contour to be considered a valid object.
# You can adjust this via the "Min Area" slider while running the app.
MIN_CONTOUR_AREA = 100  # Lowered from 300 to detect smaller objects
MAX_CONTOUR_AREA = 50000

# HSV Color Range for detecting WATER.
# You can look up "HSV Color Picker" online to find values for your specific water color.
# [Hue, Saturation, Value]
LOWER_WATER = np.array([80, 20, 20])   # Lower bound (Blue-ish / Green-ish)
UPPER_WATER = np.array([140, 255, 255]) # Upper bound

# -----------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------------------
def nothing(x):
    """ Dummy function for trackbar callbacks. """
    pass

# -----------------------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Object detection on water via clustering.")
parser.add_argument("source", nargs="?", default="kayak video on lake.mp4", 
                    help="Path to video file (default: kayak video on lake.mp4) or webcam index.")
args = parser.parse_args()

# Determine if source is a webcam index (int) or a file path (str)
try:
    source = int(args.source)
except ValueError:
    source = args.source
    if not os.path.exists(source):
        print(f"Error: Video file '{source}' not found.")
        sys.exit(1)

# -----------------------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------------------
cap = cv2.VideoCapture(source)

# If using webcam, force resolution. (Files usually ignore this).
if isinstance(source, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Create Background Subtractor
# This learns the "static" background over time to detect moving pixels.
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,        # How many frames to remember for background model
    varThreshold=20,    # Sensitivity (Lowered from 25 to detect more motion)
    detectShadows=False # Shadows can cause false positives, so we disable them
)

# Morphological Kernels
# Used for "Area Closing" and "Opening" to remove noise.
kernel = np.ones((5, 5), np.uint8)              # Small kernel for noise removal
large_kernel = np.ones((25, 25), np.uint8)      # Large kernel for defining the "Water Region"

consecutive_frames = 0
alert_end_time = 0.0 # Time (in seconds) when the alert should stop displaying
smoothed_angle = 0.0 # For the Horizon Gimbal

# -----------------------------------------------------------------------------------------
# REMOVED MANUAL CONTROLS
# The system now automatically calibrates for water color.
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -----------------------------------------------------------------------------------------
while True:
    # 1. Set Default Parameters (No more sliders)
    min_area = MIN_CONTOUR_AREA
    max_area = MAX_CONTOUR_AREA
    sensitivity = 25 # Default sensitivity
    
    fgbg.setVarThreshold(sensitivity)

    # 2. Read Frame
    ret, frame = cap.read()
    if not ret:
        # Loop video if it is a file (not webcam)
        if isinstance(source, str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break # End of webcam stream

    # Resize for consistent processing speed
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ---------------------------------------------------------------------------------
    # HORIZON LEVELING GIMBAL
    # Detects the horizon line and rotates the frame to keep it level.
    # ---------------------------------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
    
    avg_angle = 0.0
    valid_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Filter: We only care about lines that are mostly horizontal (+/- 15 deg)
            if abs(angle) < 15:
                avg_angle += angle
                valid_lines += 1
    
    if valid_lines > 0:
        avg_angle /= valid_lines
        # Smooth the angle to prevent jitter (Fast response for stronger effect)
        # 0.2 alpha = more responsive than 0.1
        smoothed_angle = 0.8 * smoothed_angle + 0.2 * avg_angle
    else:
        # If no horizon found, slowly return to 0
        smoothed_angle = 0.9 * smoothed_angle
        
    # Rotate the Frame
    M = cv2.getRotationMatrix2D((FRAME_WIDTH // 2, FRAME_HEIGHT // 2), smoothed_angle, 1.10) # 1.10 Zoom to hide corners
    frame = cv2.warpAffine(frame, M, (FRAME_WIDTH, FRAME_HEIGHT))

    # =====================================================================================
    # STEP 1: Motion Detection
    # =====================================================================================
    # Use MOG2 to find pixels that have changed compared to history.
    motion_mask = fgbg.apply(frame)
    # Remove small white noise specs (Opening: Erosion followed by Dilation)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    # Thicken the moving areas slightly (Dilation) to merge nearby pixels
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    # =====================================================================================
    # STEP 2: Water Detection (Environment Masking)
    # =====================================================================================
    # Convert to HSV color space for better color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------------------------------------------
    # AUTOMATIC WATER CALIBRATION
    # Assumption: The bottom 20% of the screen is almost always water.
    # We sample this area to find the "Water Color" for this specific frame.
    # ---------------------------------------------------------------------------------
    height, width, _ = frame.shape
    roi_hsv = hsv[int(height*0.8):, :, :] # Bottom 20% ROI

    # Calculate Median Hue and Saturation of the water
    h_median = np.median(roi_hsv[:,:,0])
    s_median = np.median(roi_hsv[:,:,1])

    # Dynamic Thresholds based on the median
    # Widen the range to include shadows and glare as "Water" (Reduces false positives)
    # Hue: +/- 25 (was 15)
    # Sat: -50 / +100 (was -30 / +60)
    lower_water_dynamic = np.array([max(0, int(h_median - 25)), max(0, int(s_median - 50)), 20])
    upper_water_dynamic = np.array([min(179, int(h_median + 25)), min(255, int(s_median + 100)), 255])
    
    # Identify pixels that match the "Water" color range
    water_mask = cv2.inRange(hsv, lower_water_dynamic, upper_water_dynamic)
    
    # Create "Water Region"
    # We "Close" the water mask (Dilation -> Erosion) with a large kernel.
    # This fills in "holes" in the water (like where a boat is) effectively creating
    # a mask of "Where the water IS" vs "Where the shore/sky IS".
    water_region = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, large_kernel)

    # ---------------------------------------------------------------------------------
    # HORIZON LINE FILTER
    # To prevent the sky/treeline from being detected as water (since they color match),
    # we explicitly ignore the top 20% of the screen (Raised from 35%).
    # ---------------------------------------------------------------------------------
    horizon_limit = int(FRAME_HEIGHT * 0.20)
    water_region[0:horizon_limit, :] = 0

    # ---------------------------------------------------------------------------------
    # PATH FILTER
    # Ignore the far left and right 10% to focus on the rower's path.
    # ---------------------------------------------------------------------------------
    margin_left = int(FRAME_WIDTH * 0.10)
    margin_right = int(FRAME_WIDTH * 0.90)
    water_region[:, 0:margin_left] = 0
    water_region[:, margin_right:] = 0
    
    # Invert water mask to find "Non-Water" things (Potential Boats/Obstacles)
    non_water_mask = cv2.bitwise_not(water_mask)

    # =====================================================================================
    # STEP 3: Logic Combination
    # =====================================================================================
    # A Valid Candidate must be:
    # 1. NOT Water colored (non_water_mask)
    # 2. INSIDE the Water Region (water_region) - effectively ignoring shore/sky
    # NOTE: We removed the motion_mask requirement to detect STATIC hazards (logs, buoys)
    candidate_mask = non_water_mask
    candidate_mask = cv2.bitwise_and(candidate_mask, water_region)

    # Clean up the candidates (Merge disjoint parts of the same object)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)

    # =====================================================================================
    # STEP 4: Object Recognition (Contours)
    # =====================================================================================
    # Find outlines of the white blobs in the candidate_mask
    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,      # Retrieve only outer contours
        cv2.CHAIN_APPROX_SIMPLE # Compress horizontal/vertical segments
    )

    object_detected_this_frame = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by Size (controlled by sliders)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by Shape (Aspect Ratio) - REMOVED per user request
        # We now accept ANY shape (long logs, tall buoys, flat boats)
        
        # Valid Object Found!
        object_detected_this_frame = True

        # Draw Bounding Box (Green)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Object", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # =====================================================================================
    # ALERT SYSTEM
    # =====================================================================================
    if object_detected_this_frame:
        consecutive_frames += 1
    else:
        consecutive_frames = 0
        
    # Trigger Condition: Object seen for > 30 frames (~1 second)
    if consecutive_frames > 30:
        # Log to console slightly after it starts (once per detection)
        if consecutive_frames == 31:
            print(f"[ALERT] Object detected at {time.ctime()}!")
            
        # Keep the alert visible for 1 second from NOW
        alert_end_time = time.time() + 1
        
    # Display Logic: Show alert if we are within the display window
    if time.time() < alert_end_time:
        cv2.putText(frame, "ALERT: OBJECT DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # =====================================================================================
    # DISPLAY OUTPUT
    # =====================================================================================
    cv2.imshow("Frame", frame)
    cv2.imshow("Motion Mask", motion_mask)
    cv2.imshow("Water Region", water_region)
    cv2.imshow("Candidate Mask", candidate_mask)

    # Press 'q' or 'ESC' to quit
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()

