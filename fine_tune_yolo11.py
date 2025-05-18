from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from pathlib import Path

def fine_tune_yolo_classification(dataset_yaml, pretrained_model='yolo11m-cls.pt', epochs=50):
    """
    Fine-tune a YOLOv11 classification model on the chess piece dataset.
    
    Args:
        dataset_yaml: Path to the dataset YAML file
        pretrained_model: Name or path of the pretrained YOLO classification model
        epochs: Number of training epochs
    """
    # Load the pre-trained model
    model = YOLO(pretrained_model)
    
    # Fine-tune the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=100,  # Standard image size for classification, adjust as needed
        batch=256,   # Batch size, adjust based on your GPU memory
        device='mps',   # Use GPU if available
        project='yolo_chess_classification',
        name='fine_tuned_model',
        patience=10,  # Early stopping patience
        lr0=0.001,    # Initial learning rate
        lrf=0.01,     # Final learning rate factor
        optimizer='Adam',
        amp=True,     # Mixed precision training
    )
    
    # Evaluate the model on validation data
    val_results = model.val()
    print(f"Validation accuracy: {val_results.top1}")
    
    return model

def main():
    # Define paths
    dataset_path = './dataset/chess_piece_similarity'
    dataset_yaml = '/Users/sahilchaddha/projects/chess_recorder/yolo_chess_dataset'
    
    # Fine-tune the model
    print("Fine-tuning YOLO model...")
    model = fine_tune_yolo_classification(dataset_yaml)
    
    # Save the final model
    model.export(format='onnx')  # Save in ONNX format for deployment
    print(f"Model exported to {Path('yolo_chess_classification/fine_tuned_model/weights/best.onnx')}")

if __name__ == "__main__":
    main()
