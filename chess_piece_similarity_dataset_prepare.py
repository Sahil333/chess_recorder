import os
import argparse
import glob
import random
from PIL import Image
import yaml
import numpy as np
import csv
from pathlib import Path

def parse_annotation_line(line, class_names):
    """
    Parses a single annotation line in the expected format:
    <class_id> <x_center> <y_center> <width> <height>

    Returns:
        tuple (class_name, x_center, y_center, width, height) if valid, otherwise None.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    # Try to interpret the first token as an integer index.
    try:
        idx = int(parts[0])
        if class_names is not None and idx < len(class_names):
            cls_token = class_names[idx]
        else:
            cls_token = parts[0]
    except ValueError:
        cls_token = parts[0]

    try:
        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None

    return cls_token, x_center, y_center, w, h

def get_piece_type(class_name):
    """
    Extracts the piece type from the class name, including pawns.
    
    Args:
        class_name (str): The class name from the YOLO dataset
        
    Returns:
        tuple: (piece_type, piece_id) where piece_type is the standardized name and 
               piece_id is an integer identifier (0-4)
    """
    lower_name = class_name.lower()
    
    # Map the piece names to standardized types
    if any(piece in lower_name for piece in ['queen']):
        return 'queen', 0 
    elif any(piece in lower_name for piece in ['rook', 'elephant']):
        return 'rook', 1
    elif any(piece in lower_name for piece in ['bishop', 'camel']):
        return 'bishop', 2
    elif any(piece in lower_name for piece in ['knight', 'horse']):
        return 'knight', 3
    elif any(piece in lower_name for piece in ['pawn']):
        return 'pawn', 4
    
    return None, None

def process_image(image_path, label_path, output_dir, class_names, file_counter):
    """
    Processes one image and its associated label file, extracting promotion pieces.
    
    Args:
        image_path (str): Path to the image file
        label_path (str): Path to the corresponding label file
        output_dir (str): Directory to save cropped pieces
        class_names (list): List of class names from the data.yaml file
        file_counter (int): Counter for unique file naming
        
    Returns:
        tuple: (updated file counter, list of (image path, piece type) pairs)
    """
    piece_info = []
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}. Skipping...")
        return file_counter, piece_info

    img_width, img_height = image.size

    if not os.path.exists(label_path):
        print(f"No label file for {image_path}. Skipping...")
        return file_counter, piece_info

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        ann = parse_annotation_line(line, class_names)
        if not ann:
            continue
        class_name, x_center, y_center, w, h = ann
        
        # Get the standardized piece type (queen, rook, bishop, knight)
        piece_type, piece_id = get_piece_type(class_name)
        if not piece_type:
            continue  # Skip pieces that aren't one of our target types
        
        # Convert normalized coordinates (YOLO format) to pixel values
        box_w = w * img_width
        box_h = h * img_height
        cx = x_center * img_width
        cy = y_center * img_height

        left = int(cx - box_w / 2)
        top = int(cy - box_h / 2)
        right = int(cx + box_w / 2)
        bottom = int(cy + box_h / 2)

        # Clamp coordinates to image boundaries
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        if right <= left or bottom <= top:
            continue  # Invalid box, skip

        cropped = image.crop((left, top, right, bottom))
        
        # Create output directory for this piece type if it doesn't exist
        piece_dir = os.path.join(output_dir, piece_type)
        os.makedirs(piece_dir, exist_ok=True)
        
        # Save the cropped image
        file_name = f"piece_{file_counter:06d}.jpg"
        output_path = os.path.join(piece_dir, file_name)
        try:
            cropped.save(output_path)
            piece_info.append((output_path, piece_type, piece_id))
            file_counter += 1
        except Exception as e:
            print(f"Error saving cropped image {output_path}: {e}. Skipping...")
            continue
            
    return file_counter, piece_info

def generate_pairs(piece_info, output_dir, num_pairs_per_piece=5, same_type_ratio=0.5):
    """
    Generates pairs of images for Siamese network training.
    
    Args:
        piece_info (list): List of (image path, piece type) pairs
        output_dir (str): Output directory for pair listings
        num_pairs_per_piece (int): Number of pairs to generate per piece
        same_type_ratio (float): Ratio of same-type pairs vs different-type pairs
        
    Returns:
        None
    """
    if not piece_info:
        print("No pieces found to generate pairs.")
        return
    
    # Group pieces by type
    pieces_by_type = {}
    for path, piece_type, _ in piece_info:
        if piece_type not in pieces_by_type:
            pieces_by_type[piece_type] = []
        pieces_by_type[piece_type].append(path)
    
    # Prepare CSV files for train/val/test splits
    os.makedirs(output_dir, exist_ok=True)
    train_csv = open(os.path.join(output_dir, 'train_pairs.csv'), 'w', newline='')
    val_csv = open(os.path.join(output_dir, 'val_pairs.csv'), 'w', newline='')
    test_csv = open(os.path.join(output_dir, 'test_pairs.csv'), 'w', newline='')
    
    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)
    test_writer = csv.writer(test_csv)
    
    train_writer.writerow(['image1', 'image2', 'same_type'])
    val_writer.writerow(['image1', 'image2', 'same_type'])
    test_writer.writerow(['image1', 'image2', 'same_type'])
    
    # Generate pairs for each piece
    for path, piece_type, _ in piece_info:
        # For each piece, generate num_pairs_per_piece pairs
        for _ in range(num_pairs_per_piece):
            # Decide if this will be a same-type or different-type pair
            same_type = random.random() < same_type_ratio
            
            if same_type:
                # Find another piece of the same type
                if len(pieces_by_type[piece_type]) > 1:
                    # Exclude the current piece from potential matches
                    candidates = [p for p in pieces_by_type[piece_type] if p != path]
                    if candidates:
                        other_path = random.choice(candidates)
                    else:
                        # Not enough pieces of this type, skip
                        continue
                else:
                    # Not enough pieces of this type, skip
                    continue
            else:
                # Find a piece of a different type
                other_types = [t for t in pieces_by_type.keys() if t != piece_type]
                if not other_types:
                    # No other types available, skip
                    continue
                other_type = random.choice(other_types)
                other_path = random.choice(pieces_by_type[other_type])
            
            # Determine which split to assign this pair to
            r = random.random()
            if r < 0.7:  # 70% train
                writer = train_writer
            elif r < 0.85:  # 15% validation
                writer = val_writer
            else:  # 15% test
                writer = test_writer
            
            # Write the pair to the CSV
            writer.writerow([path, other_path, int(same_type)])
    
    # Close all CSV files
    train_csv.close()
    val_csv.close()
    test_csv.close()
    
    print(f"Pair generation complete. Files saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for training a Siamese neural network to identify similar chess pieces."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input YOLO dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output dataset directory")
    parser.add_argument("--pairs_per_piece", type=int, default=5,
                        help="Number of pairs to generate per piece (default: 5)")
    parser.add_argument("--same_type_ratio", type=float, default=0.5,
                        help="Ratio of same-type pairs vs different-type pairs (default: 0.5)")
    args = parser.parse_args()

    # Load the data.yaml file from the main input directory
    data_yaml_path = os.path.join(args.input_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        print(f"data.yaml not found in input directory: {data_yaml_path}. Exiting...")
        return

    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    if "names" not in data_yaml:
        print("No 'names' field found in data.yaml. Exiting...")
        return

    class_names = data_yaml["names"]
    
    # Create output directories
    cropped_dir = os.path.join(args.output_dir, "cropped_pieces")
    pairs_dir = os.path.join(args.output_dir, "pairs")
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(pairs_dir, exist_ok=True)
    
    # Process all images
    sub_dirs = ["train", "valid", "test"]
    file_counter = 0
    all_piece_info = []
    
    for sub in sub_dirs:
        cur_images_dir = os.path.join(args.input_dir, sub, "images")
        cur_labels_dir = os.path.join(args.input_dir, sub, "labels")
        
        if not os.path.exists(cur_images_dir) or not os.path.exists(cur_labels_dir):
            print(f"Images or labels directory not found for '{sub}' split. Skipping...")
            continue
        
        image_files = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            image_files.extend(glob.glob(os.path.join(cur_images_dir, ext)))
        
        if not image_files:
            print(f"No images found in {cur_images_dir}. Skipping '{sub}' split.")
            continue
        
        for image_file in image_files:
            base = os.path.basename(image_file)
            file_root, _ = os.path.splitext(base)
            label_file = os.path.join(cur_labels_dir, file_root + ".txt")
            
            file_counter, piece_info = process_image(
                image_file, label_file, cropped_dir, class_names, file_counter
            )
            all_piece_info.extend(piece_info)

    # Save the piece info as a separate dataset path - randomized and split 80% train, 10% val, 10% test in a csv file with piece info and piece type
    train_info_path = os.path.join(args.output_dir, "train_pieces.csv")
    val_info_path = os.path.join(args.output_dir, "val_pieces.csv")
    test_info_path = os.path.join(args.output_dir, "test_pieces.csv")

    # Randomize the piece info
    random.shuffle(all_piece_info)
    
    # Split the piece info into train, val, test
    train_piece_info = all_piece_info[:int(len(all_piece_info) * 0.8)]
    val_piece_info = all_piece_info[int(len(all_piece_info) * 0.8):int(len(all_piece_info) * 0.9)]
    test_piece_info = all_piece_info[int(len(all_piece_info) * 0.9):]
    
    # Save the piece info to the csv files
    with open(train_info_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "piece_id"])
        for info in train_piece_info:
            writer.writerow([info[0], info[2]])
    
    with open(val_info_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "piece_id"])
        for info in val_piece_info:
            writer.writerow([info[0], info[2]])
    
    with open(test_info_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "piece_id"])
        for info in test_piece_info:
            writer.writerow([info[0], info[2]])

    
    # Generate pairs for Siamese network training
    generate_pairs(
        all_piece_info, 
        pairs_dir, 
        num_pairs_per_piece=args.pairs_per_piece,
        same_type_ratio=args.same_type_ratio
    )
    
    print(f"Dataset preparation completed. Total cropped pieces: {file_counter}")
    
    # Create summary file with dataset statistics
    piece_types = {info[1] for info in all_piece_info}
    piece_counts = {piece_type: sum(1 for _, pt, _ in all_piece_info if pt == piece_type) 
                   for piece_type in piece_types}
    
    with open(os.path.join(args.output_dir, "dataset_summary.txt"), "w") as f:
        f.write("Chess Piece Similarity Dataset Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Total pieces extracted: {file_counter}\n\n")
        f.write("Pieces by type:\n")
        for piece_type, count in piece_counts.items():
            f.write(f"- {piece_type}: {count}\n")

if __name__ == "__main__":
    main()
