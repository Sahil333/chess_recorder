from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11m.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/Users/sahilchaddha/projects/chess recorder/dataset/Chess Piece Dataset.v1i.yolov11/data.yaml", epochs=100, imgsz=640, device="mps")
results.show()
results = model.val("/Users/sahilchaddha/projects/chess recorder/dataset/Chess Piece Dataset.v1i.yolov11/test/images", save=True)
