from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("runs/detect/train/weights/last.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.predict("/Users/sahilchaddha/Downloads/WhatsApp Image 2025-02-12 at 03.49.20.jpeg", save=True)
