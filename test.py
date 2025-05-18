from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("runs/detect/train/weights/last.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.predict("/Users/sahilchaddha/projects/chess_recorder/debug_frame_20250216_065800_355.jpg", save=True)
