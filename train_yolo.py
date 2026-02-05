from ultralytics import YOLO

def main():
    # 1. Load a model
    # We use 'yolov8n.pt' which is the Nano version (fastest, smallest)
    # It will automatically download the weights if they don't exist
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    # data: Path to your data.yaml file
    # epochs: Number of training cycles (100 is a good start)
    # imgsz: Image size (640 is standard for YOLO)
    # batch: Batch size (16 is usually safe for most GPUs, lower to 8 or 4 if you run out of memory)
    # device: '0' for GPU, 'cpu' for CPU (Auto-detects usually)
    results = model.train(
        data=r'c:\Users\joshm\Senior Design\datasets\marine_debris\data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='marine_debris_model', # This creates runs/detect/marine_debris_model
        patience=50                 # Stop early if no improvement for 50 epochs
    )

if __name__ == '__main__':
    # Using this guard because multiprocessing (used by data loaders) requires it on Windows
    main()
