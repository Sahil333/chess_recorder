import os

# Paths to the dataset
train_label_dir = "/Users/sahilchaddha/projects/chess recorder/dataset/Chess Piece Dataset.v1i.yolov11/train/labels"  # Update with your path to 'train/labels' or 'val/labels'
val_label_dir = "/Users/sahilchaddha/projects/chess recorder/dataset/Chess Piece Dataset.v1i.yolov11/valid/labels"  # Update with your path to 'train/labels' or 'val/labels'
test_label_dir = "/Users/sahilchaddha/projects/chess recorder/dataset/Chess Piece Dataset.v1i.yolov11/test/labels"  # Update with your path to 'train/labels' or 'val/labels'

# Define the new class ID for "chess_piece"
NEW_CLASS_ID = 0  # YOLO class IDs are zero-indexed

def convert_labels(label_dir):
    # Iterate over all label files in the directory
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)
            
            with open(label_path, "r") as file:
                lines = file.readlines()
            
            # Modify each line to use the new class ID
            new_lines = []
            for line in lines:
                _, x_center, y_center, width, height = line.strip().split()
                new_lines.append(f"{NEW_CLASS_ID} {x_center} {y_center} {width} {height}\n")
            
            # Write the updated labels back to the file
            with open(label_path, "w") as file:
                file.writelines(new_lines)

# Convert labels for the dataset
convert_labels(train_label_dir)
convert_labels(val_label_dir)
convert_labels(test_label_dir)
print(f"All labels in '{label_dir}' have been converted to the 'chess_piece' class.")
