import os
import argparse
import torch
import numpy as np
import tensorflow as tf

def convert_ultralytics_yolo_to_tflite(model_path, output_dir):
    """
    Convert a YOLO model to TFLite using Ultralytics built-in export
    
    Args:
        model_path: Path to the YOLO model (.pt)
        output_dir: Directory to save the output models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import the YOLO class from ultralytics
        from ultralytics import YOLO
        
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Export the model to TFLite format
        # By default this creates a file in the same directory as the model
        exported_path = model.export(format="tflite", imgsz=640)
        
        # The export() method returns the path to the exported model
        print(f"TFLite model exported to: {exported_path}")
        
        # Optionally copy to the output_dir if different from current directory
        import shutil
        base_name = os.path.basename(exported_path)
        output_path = os.path.join(output_dir, base_name)
        
        if exported_path != output_path:
            shutil.copy(exported_path, output_path)
            print(f"Copied TFLite model to: {output_path}")
        
        return output_path
        
    except ImportError:
        print("Ultralytics package not found. Please install with: pip install ultralytics")
        return None

def convert_pytorch_yolo_to_tflite(model_path, output_dir, input_shape=(1, 3, 640, 640)):
    """
    Convert a PyTorch YOLO model to TFLite format via ONNX
    
    Args:
        model_path: Path to the PyTorch model (.pt)
        output_dir: Directory to save the output models
        input_shape: Shape of the input tensor (batch_size, channels, height, width)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import onnx
        import onnx_tf
        
        # Step 1: Load PyTorch model - this is a simplified example
        # You'll need to adapt this to your specific YOLO model structure
        device = torch.device("cpu")
        
        # Load the model - this is a placeholder, adapt to your model's loading method
        model = torch.load(model_path, map_location=device)
        if hasattr(model, 'eval'):  # If it's a nn.Module
            model.eval()
        elif isinstance(model, dict) and 'model' in model:  # If it's a state dict
            from ultralytics.nn.tasks import attempt_load_one_weight
            model = attempt_load_one_weight(model_path)
            model.eval()
        
        # Create a dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Step 2: Export to ONNX
        onnx_path = os.path.join(output_dir, "yolo_model.onnx")
        torch.onnx.export(model, 
                        dummy_input, 
                        onnx_path,
                        export_params=True,
                        opset_version=12,  # YOLO may need a newer opset
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
        
        print(f"ONNX model saved to {onnx_path}")
        
        # Step 3: Load ONNX model and verify
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model checked successfully")
        
        # Step 4: Convert ONNX to TensorFlow
        tf_path = os.path.join(output_dir, "tf_model")
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_rep.export_graph(tf_path)
        print(f"TensorFlow model saved to {tf_path}")
        
        # Step 5: Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_path = os.path.join(output_dir, "yolo_model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {tflite_path}")
        
        return tflite_path
        
    except ImportError as e:
        print(f"Error importing required packages: {e}")
        print("Make sure to install required packages: pip install onnx onnx-tf")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO model to TFLite")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the YOLO model (.pt)")
    parser.add_argument("--output_dir", type=str, default="converted_models",
                        help="Directory to save the output models")
    parser.add_argument("--method", type=str, choices=["ultralytics", "pytorch"], default="ultralytics",
                        help="Conversion method to use")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for the model input (square)")
    args = parser.parse_args()
    
    print(f"TensorFlow version: {tf.__version__}")
    
    if args.method == "ultralytics":
        print("Using Ultralytics YOLO export method...")
        output_path = convert_ultralytics_yolo_to_tflite(args.model_path, args.output_dir)
    else:
        print("Using PyTorch to ONNX to TFLite conversion path...")
        input_shape = (1, 3, args.img_size, args.img_size)
        output_path = convert_pytorch_yolo_to_tflite(args.model_path, args.output_dir, input_shape)
    
    if output_path:
        print(f"\nConversion completed successfully!")
        print(f"TFLite model: {output_path}")
    else:
        print("\nConversion failed. Please check the error messages above.") 