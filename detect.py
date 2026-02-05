from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Perform object detection on an image using the model
results = model(source='0', show=True, conf=0.5)  # accept all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
