import cv2
import numpy as np
import argparse
import sys
import os

# -----------------------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------------------
# Dimensions to resize the video to. Smaller = Faster processing.
FRAME_WIDTH = 640
FRAME_HEIGHT = 384

# Minimum area (pixels) for a contour to be considered a valid object.
# You can adjust this via the "Min Area" slider while running the app.
MIN_CONTOUR_AREA = 300  
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
parser.add_argument("source", nargs="?", default="boatsingle.mp4", 
                    help="Path to video file (default: boatsingle.mp4) or webcam index.")
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
    varThreshold=25,    # Sensitivity (Adjustable via slider)
    detectShadows=False # Shadows can cause false positives, so we disable them
)

# Morphological Kernels
# Used for "Area Closing" and "Opening" to remove noise.
kernel = np.ones((5, 5), np.uint8)              # Small kernel for noise removal
large_kernel = np.ones((25, 25), np.uint8)      # Large kernel for defining the "Water Region"

# -----------------------------------------------------------------------------------------
# GUI CONTROLS
# -----------------------------------------------------------------------------------------
cv2.namedWindow("Controls")
# Trackbar: Filter out small noise (waves)
cv2.createTrackbar("Min Area", "Controls", MIN_CONTOUR_AREA, 5000, nothing)
# Trackbar: Filter out huge objects (if needed)
cv2.createTrackbar("Max Area", "Controls", MAX_CONTOUR_AREA // 100, 1000, nothing) 
# Trackbar: Adjust motion sensitivity (Lower = Detect more validity)
cv2.createTrackbar("Sensitivity", "Controls", 25, 100, nothing) 

# -----------------------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -----------------------------------------------------------------------------------------
while True:
    # 1. Update Parameters from Trackbars
    min_area = cv2.getTrackbarPos("Min Area", "Controls")
    max_area = cv2.getTrackbarPos("Max Area", "Controls") * 100
    sensitivity = cv2.getTrackbarPos("Sensitivity", "Controls")
    
    fgbg.setVarThreshold(sensitivity)

    # 2. Read Frame
    ret, frame = cap.read()
    if not ret:
        break # End of video

    # Resize for consistent processing speed
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

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
    
    # Identify pixels that match the "Water" color range
    water_mask = cv2.inRange(hsv, LOWER_WATER, UPPER_WATER)
    
    # Create "Water Region"
    # We "Close" the water mask (Dilation -> Erosion) with a large kernel.
    # This fills in "holes" in the water (like where a boat is) effectively creating
    # a mask of "Where the water IS" vs "Where the shore/sky IS".
    water_region = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, large_kernel)
    
    # Invert water mask to find "Non-Water" things (Potential Boats/Obstacles)
    non_water_mask = cv2.bitwise_not(water_mask)

    # =====================================================================================
    # STEP 3: Logic Combination
    # =====================================================================================
    # A Valid Candidate must be:
    # 1. Moving (motion_mask)
    # 2. NOT Water colored (non_water_mask)
    # 3. INSIDE the Water Region (water_region) - effectively ignoring shore/sky
    candidate_mask = cv2.bitwise_and(motion_mask, non_water_mask)
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

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by Size (controlled by sliders)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by Shape (Aspect Ratio)
        # We reject extremely wide/flat objects (often just wave lines)
        aspect_ratio = w / float(h)
        if aspect_ratio > 8 or aspect_ratio < 0.1:
            continue

        # Draw Bounding Box (Green)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Object", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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

