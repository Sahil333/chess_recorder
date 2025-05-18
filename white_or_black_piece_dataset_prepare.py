import os
import argparse
import glob
import random
from PIL import Image
import yaml

def parse_annotation_line(line, class_names):
    """
    Parses a single annotation line in the expected format:
    <class_name> <x_center> <y_center> <width> <height>

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

def process_image(image_path, label_path, output_base, ratios, file_counter, class_names):
    """
    Processes one image and its associated label file. For each detection,
    the function determines the piece color from the annotation (by checking
    if the class name contains 'white' or 'black'). It then crops the piece,
    assigns it randomly to a train/val/test split (using provided ratios), and
    writes out the cropped image and a label file. 

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the corresponding label file.
        output_base (str): Base path for the output dataset.
        ratios (tuple): Tuple of (train_ratio, val_ratio, test_ratio).
        file_counter (int): A counter used to name the cropped piece files uniquely.
        class_names (list): List of class names from the data.yaml file.
        
    Returns:
        int: Updated file_counter.
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}. Skipping...")
        return file_counter

    img_width, img_height = image.size

    if not os.path.exists(label_path):
        print(f"No label file for {image_path}. Skipping...")
        return file_counter

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        ann = parse_annotation_line(line, class_names)
        if not ann:
            continue
        class_name, x_center, y_center, w, h = ann

        lower_name = class_name.lower()
        if "white" in lower_name:
            new_class = 0  # white_piece
        elif "black" in lower_name:
            new_class = 1  # black_piece
        else:
            # Skip detections for which we cannot determine the piece color
            continue
        
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
        
        # Randomly assign this cropped image to a split based on the provided ratios
        r = random.random()
        train_ratio, val_ratio, _ = ratios
        if r < train_ratio:
            split = "train"
        elif r < train_ratio + val_ratio:
            split = "valid"
        else:
            split = "test"
        
        # Create the output directories if not already present
        images_dir = os.path.join(output_base, split, "images")
        labels_dir = os.path.join(output_base, split, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Generate a unique file name based on file_counter
        base_filename = f"piece_{file_counter:06d}"
        file_counter += 1

        image_output_path = os.path.join(images_dir, base_filename + ".jpg")
        label_output_path = os.path.join(labels_dir, base_filename + ".txt")
        
        # Save the cropped image as the new piece image
        try:
            cropped.save(image_output_path)
        except Exception as e:
            print(f"Error saving cropped image {image_output_path}: {e}. Skipping...")
            continue

        # Since the cropped piece occupies the entire image, the annotation becomes:
        # <new_class> 0.5 0.5 1.0 1.0  (class, center, width, height)
        with open(label_output_path, "w") as f_out:
            f_out.write(f"{new_class} 0.5 0.5 1.0 1.0\n")
            
    return file_counter

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a two-class (white_piece, black_piece) dataset from a YOLO dataset."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input YOLO dataset directory containing 'train', 'valid', and 'test' subdirectories (and a data.yaml file) with 'images' and 'labels' folders.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output dataset directory.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training split ratio. (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Valid split ratio. (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio. (default: 0.15)")
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

    sum_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(sum_ratio - 1.0) > 1e-6:
        print("Error: train_ratio, val_ratio, and test_ratio must sum to 1.0")
        return

    sub_dirs = ["train", "valid", "test"]
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    file_counter = 0
    for sub in sub_dirs:
        cur_images_dir = os.path.join(args.input_dir, sub, "images")
        cur_labels_dir = os.path.join(args.input_dir, sub, "labels")
        
        if not os.path.exists(cur_images_dir):
            print(f"Input images directory not found: {cur_images_dir}. Skipping '{sub}' split.")
            continue
        if not os.path.exists(cur_labels_dir):
            print(f"Input labels directory not found: {cur_labels_dir}. Skipping '{sub}' split.")
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
            file_counter = process_image(image_file, label_file, args.output_dir, ratios, file_counter, class_names)
    
    print("Dataset preparation completed. Total cropped pieces:", file_counter)

if __name__ == "__main__":
    main()
