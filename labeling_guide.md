# How to Label and Train on Your Own Images

To make the AI detect new objects (like rocks, logs, or specific hazards), you need to "teach" it by labeling photos.

## Step 1: Collect Photos
1.  **Take Photos**: Capture images of the hazards you want to detect.
    *   *Tip*: Take photos from different angles, distances, and lighting conditions (sunny, cloudy).
    *   *Tip*: Include "negative" images (water with *no* hazards) to reduce false alarms.
2.  **Organize**: Put all your raw `.jpg` or `.png` images in a folder on your computer.

## Step 2: Label Images (Draw Boxes)
You need a tool to draw boxes around the objects. I recommend **Roboflow** (easiest, online) or **LabelImg** (local).

### Option A: Using Roboflow (Recommended)
1.  Go to [roboflow.com](https://roboflow.com) and create a free account.
2.  Create a **New Project** (type: "Object Detection").
3.  **Upload** your photos.
4.  **Annotate**: Click and drag to draw boxes around your objects.
    *   Name the classes clearly (e.g., `rock`, `log`, `buoy`).
5.  **Generate Version**: Click "Generate" -> "Export".
6.  **Format**: Select **YOLOv8** format.
7.  **Download**: Choose "Zip to computer".

### Option B: Using LabelImg (Local)
1.  Install: `pip install labelImg` then run `labelImg`.
2.  Open your image folder.
3.  Draw boxes (Press `w` to draw).
4.  Save: Ensure format is set to **YOLO** (not PascalVOC). It will save `.txt` files next to your images.

## Step 3: Import to VS Code
Once you have your downloaded dataset (images `.jpg` and labels `.txt`):

1.  **Clear Old Data** (Optional): If you want to start fresh, delete contents of `datasets/marine_debris/data/train` and `valid`.
2.  **Copy Files**:
    *   Put **Training** images in: `datasets/marine_debris/data/train/images/`
    *   Put **Training** labels in: `datasets/marine_debris/data/train/labels/`
    *   Put **Validation** images in: `datasets/marine_debris/data/valid/images/` (put about 20% of your photos here)
    *   Put **Validation** labels in: `datasets/marine_debris/data/valid/labels/`

## Step 4: Update Configuration
1.  Open `datasets/marine_debris/data.yaml`.
2.  Update the **names** list to match your new labels.
    *   *Important*: The names must be in the exact order as your annotation tool gave them (usually alphabetical or creation order).

    ```yaml
    # Example data.yaml
    path: C:/Users/joshm/Senior Design/datasets/marine_debris/data
    train: train/images
    val: valid/images

    names:
      0: rock
      1: log
      2: buoy
    ```

## Step 5: Train Again
Run the training command in the terminal:

```powershell
.venv\Scripts\yolo detect train model=yolov8n.pt data="datasets/marine_debris/data.yaml" epochs=100 imgsz=640
```
