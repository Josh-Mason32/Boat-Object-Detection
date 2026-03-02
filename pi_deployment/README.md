# Boat Object Detection (Raspberry Pi Deployment)

This directory contains standalone files needed to run the YOLOv8 object detection on a Raspberry Pi.

## Setup

1. Copy this folder to your Raspberry Pi.
2. Open a terminal in this folder on the Pi.
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ultralytics installation might take a while on a Pi depending on your internet connection. We recommend setting up a virtual environment `python3 -m venv venv` first).*

## Running the Prediction

You can run the script in two ways:

### 1. Using a Live Camera
By default, if you don't provide a video file path, it will try to use a camera connected to the Pi (index `0`):
```bash
python predict_poc.py
```

### 2. Using a Video File
If you have a video file on the Pi, you can pass its path as an argument:
```bash
python predict_poc.py path/to/video.mp4
```

## Files in this Folder
- `predict_poc.py` : The customized detection script (updated so it looks for `best.pt` in the same directory).
- `best.pt` : The trained custom YOLOv8 model weights.
- `requirements.txt` : The necessary Python packages.
