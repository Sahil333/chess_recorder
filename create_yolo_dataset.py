import os
import shutil
import pandas as pd
import yaml
from pathlib import Path

def create_yolo_classification_dataset(dataset_path, output_dir):
    """
    Create a YOLO-compatible classification dataset from the existing chess piece CSV files
    
    Args:
        dataset_path: Path to the directory containing train_pieces.csv, val_pieces.csv, etc.
        output_dir: Path where the YOLO-formatted dataset will be created
    """
    # Define class names (update these based on your actual classes)
    class_names = ['queen', 'rook', 'bishop', 'knight', 'pawn']
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(dataset_path, f'{split}_pieces.csv')
        
        # Skip if file doesn't exist
        if not os.path.exists(csv_path):
            print(f"Skipping {split} - file not found: {csv_path}")
            continue
            
        # Create split directory
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class directories
        for class_name in class_names:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Read CSV file
        pieces_df = pd.read_csv(csv_path)
        print(f"Processing {len(pieces_df)} images for {split} split")
        
        # Process each row
        for _, row in pieces_df.iterrows():
            img_path = row.iloc[0]
            class_idx = int(row.iloc[1])
            
            # Ensure the class index is valid
            if class_idx < 0 or class_idx >= len(class_names):
                print(f"Warning: Invalid class index {class_idx} for {img_path}")
                continue
                
            # Get the full source path
            # if not os.path.isabs(img_path):
            #     src_path = os.path.join(dataset_path, img_path)
            # else:
            src_path = img_path
                
            # Create destination path
            class_name = class_names[class_idx]
            dst_path = os.path.join(split_dir, class_name, os.path.basename(img_path))
            
            # Copy or symlink the file
            if os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    # Use symlink to save space (or copy if needed)
                    try:
                        os.symlink(os.path.abspath(src_path), dst_path)
                    except OSError:
                        # Fallback to copy if symlink fails
                        shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Source file not found: {src_path}")
    
    # Create YAML configuration file
    dataset_config = {
        'path': os.path.abspath(output_dir),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
        
    print(f"YOLO dataset created at: {output_dir}")
    print(f"Dataset configuration saved to: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    create_yolo_classification_dataset(
        dataset_path='./dataset/chess_piece_similarity',
        output_dir='./yolo_chess_dataset'
    ) 