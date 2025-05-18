import torch
import numpy as np
import os
import argparse
from white_or_black_classifier import SimpleCNN
import tensorflow as tf

def convert_direct_to_tflite(model_path, output_dir, input_shape=(1, 3, 75, 75)):
    """
    Convert a PyTorch model directly to TFLite by recreating in TensorFlow
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load PyTorch model
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Step 2: Create a dummy input and get the output
    dummy_input = torch.randn(input_shape, device=device)
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()
    
    # Step 3: Create equivalent TensorFlow model
    from tensorflow import keras

    tf_model = keras.Sequential([
        keras.layers.Input(shape=(3, 75, 75)),  # NCHW format (PyTorch)
        keras.layers.Permute((2, 3, 1)),        # Convert NCHW to NHWC
        keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(2)
    ])
    
    # Transfer weights manually (this is a simplified example)
    # In a real scenario, you'd need to carefully map each layer's weights
    
    # For demonstration - compile the model
    tf_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the TF model
    tf_path = os.path.join(output_dir, "tf_model")
    tf_model.save(tf_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_path = os.path.join(output_dir, "direct_white_black_classifier.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_path}")
    
    return tflite_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model directly to TFLite")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="converted_models")
    args = parser.parse_args()
    
    tflite_path = convert_direct_to_tflite(args.model_path, args.output_dir)
    print(f"Conversion complete. Model saved to {tflite_path}") 