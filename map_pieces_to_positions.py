import numpy as np
from typing import List, Tuple
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt
import argparse
from shapely.geometry import Polygon

def detect_pieces(image):
    from ultralytics import YOLO

    # Load the trained YOLO model
    model = YOLO("runs/detect/train/weights/last.pt")

    # Run inference on the image
    results = model.predict(image)
    return results[0]  # Return the first (and only) result

def detect_board(image):
    from chesscog.corner_detection.detect_corners import find_corners
    from recap import URI, CfgNode as CN

    # Set random seed for reproducibility
    # np.random.seed(42)
    
    cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
    
    # Modify config to increase stability if needed
    cfg.defrost()
    cfg.RANSAC.BEST_SOLUTION_TOLERANCE = 0.1  # Make it more strict
    cfg.RANSAC.OFFSET_TOLERANCE = 0.1  # Make it more strict
    cfg.freeze()
    
    corners, homography = find_corners(cfg, image)
    return corners, homography

def determine_board_orientation(warped_boxes, chess_squares, warped_image):
    """Determine board orientation by analyzing piece colors"""
    # Split boxes into left and right halves
    mid_x = (warped_image.shape[1]) / 2
    left_boxes = []
    right_boxes = []
    
    for box in warped_boxes:
        box_center_x = (box[0] + box[2]) / 2
        if box_center_x < mid_x:
            left_boxes.append(box)
        else:
            right_boxes.append(box)
    
    # Calculate average intensity for pieces on each side
    def calculate_avg_intensity(boxes):
        if not boxes:
            return 0
        intensities = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            piece_region = warped_image[y1:y2, x1:x2]
            # Convert to grayscale if image is in color
            if len(piece_region.shape) == 3:
                piece_region = cv2.cvtColor(piece_region, cv2.COLOR_BGR2GRAY)
            avg_intensity = np.mean(piece_region)
            intensities.append(avg_intensity)
        return np.mean(intensities)
    
    left_intensity = calculate_avg_intensity(left_boxes)
    right_intensity = calculate_avg_intensity(right_boxes)
    
    # Lower intensity means darker pieces (black)
    black_on_left = left_intensity < right_intensity
    
    print(f"Left intensity: {left_intensity:.2f}, Right intensity: {right_intensity:.2f}")
    print(f"Black pieces are on the {'left' if black_on_left else 'right'} side")
    
    # If black is on right, 'a' starts from top (no flip needed)
    # If black is on left, 'a' starts from bottom (need to flip)
    needs_flip = black_on_left
    return needs_flip

def create_chess_grid(warped_corners, needs_flip=False):
    """Create 8x8 grid coordinates from warped corner points"""
    # Sort corners to ensure order: top-left, top-right, bottom-right, bottom-left
    x_coords = np.linspace(warped_corners[0][0], warped_corners[1][0], 9)
    y_coords = np.linspace(warped_corners[0][1], warped_corners[3][1], 9)
    
    # Create grid squares
    squares = []
    for i in range(8):
        for j in range(8):
            square = np.array([
                [x_coords[j], y_coords[i]],      # top-left
                [x_coords[j+1], y_coords[i]],    # top-right
                [x_coords[j+1], y_coords[i+1]],  # bottom-right
                [x_coords[j], y_coords[i+1]]     # bottom-left
            ])
            # If black is on right, 'a' starts from top
            # If black is on left, 'a' starts from bottom
            if needs_flip:
                square_name = f"{chr(97 + (7-i))}{j+1}"  # a1 at bottom
            else:
                square_name = f"{chr(97 + i)}{8-j}"  # a8 at top
            squares.append((square, square_name))
    return squares

def calculate_overlap(box, square):
    """Calculate overlap area between a box and a square"""
    # Convert box and square to polygons
    box_poly = Polygon(box)
    square_poly = Polygon(square)
    
    if not box_poly.intersects(square_poly):
        return 0
    
    return box_poly.intersection(square_poly).area / box_poly.area

def map_pieces_to_positions(image):
    # Detect the board and get homography matrix
    corners, homography = detect_board(image)    
    
    # Create a copy of original image to draw original corners
    original_with_corners = image.copy()
    for i, corner in enumerate(corners):
        cv2.circle(original_with_corners, (int(corner[0]), int(corner[1])), 
                  5, (0, 255, 0), -1)
        cv2.putText(original_with_corners, str(i+1), 
                   (int(corner[0])+10, int(corner[1])+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # use the same output size as the original image
    output_size = image.shape[1::-1]
    
    # Apply perspective transformation
    warped_image = cv2.warpPerspective(image, homography, output_size)
    
    # Convert corners to homogeneous coordinates
    corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
    
    # Transform corners using homography matrix
    warped_corners = homography @ corners_homogeneous.T
    # Convert back from homogeneous coordinates
    warped_corners = warped_corners / warped_corners[2]
    warped_corners = warped_corners[:2].T
    
    # Detect the pieces in the image
    results = detect_pieces(image)
    boxes = results.boxes.xyxy.cpu().numpy()
    
    # Transform bounding boxes to warped space
    warped_boxes = []
    for box in boxes:
        # Get the corners of the bounding box
        box_corners = np.array([
            [box[0], box[1]],  # top-left
            [box[2], box[1]],  # top-right
            [box[2], box[3]],  # bottom-right
            [box[0], box[3]]   # bottom-left
        ])
        
        # Convert to homogeneous coordinates
        box_corners_homogeneous = np.hstack([box_corners, np.ones((4, 1))])
        
        # Transform using homography
        warped_box_corners = homography @ box_corners_homogeneous.T
        warped_box_corners = warped_box_corners / warped_box_corners[2]
        warped_box_corners = warped_box_corners[:2].T
        
        # Calculate new bounding box from transformed corners
        warped_box = [
            np.min(warped_box_corners[:, 0]),  # x1
            np.min(warped_box_corners[:, 1]),  # y1
            np.max(warped_box_corners[:, 0]),  # x2
            np.max(warped_box_corners[:, 1])   # y2
        ]
        warped_boxes.append(warped_box)
    warped_boxes = np.array(warped_boxes)

    # Create initial chess grid
    chess_squares = create_chess_grid(warped_corners)
    
    # Determine correct orientation based on piece colors
    needs_flip = determine_board_orientation(warped_boxes, chess_squares, warped_image)
    
    # Recreate chess grid with correct orientation
    chess_squares = create_chess_grid(warped_corners, needs_flip)
    
    # Map pieces to squares
    piece_positions = []
    for box in warped_boxes:
        # Create bottom half of the bounding box
        mid_y = (box[1] + box[3]) / 2
        box_bottom_half = np.array([
            [box[0], mid_y],    # top-left
            [box[2], mid_y],    # top-right
            [box[2], box[3]],   # bottom-right
            [box[0], box[3]]    # bottom-left
        ])
        
        # Find square with maximum overlap
        max_overlap = 0
        best_square = None
        for square_coords, square_name in chess_squares:
            overlap = calculate_overlap(box_bottom_half, square_coords)
            if overlap > max_overlap:
                max_overlap = overlap
                best_square = square_name
        
        if best_square:
            piece_positions.append((best_square, box))

    # For visualization, let's draw both corners, boxes and positions
    visualization_image = warped_image.copy()
    
    # Draw corners
    for i, corner in enumerate(warped_corners):
        cv2.circle(visualization_image, (int(corner[0]), int(corner[1])), 
                  5, (0, 255, 0), -1)
    
    # Draw chess grid
    for square_coords, _ in chess_squares:
        for i in range(4):
            pt1 = tuple(map(int, square_coords[i]))
            pt2 = tuple(map(int, square_coords[(i + 1) % 4]))
            cv2.line(visualization_image, pt1, pt2, (0, 255, 0), 1)
    
    # Draw boxes and positions
    for square_name, box in piece_positions:
        cv2.rectangle(visualization_image, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (0, 0, 255), 2)
        # Put square name at center of box
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        cv2.putText(visualization_image, square_name, 
                   (center_x, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return original_with_corners, visualization_image, corners, warped_corners, boxes, warped_boxes, piece_positions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chess board detection and piece mapping')
    parser.add_argument('--image', type=str, 
                       default="/Users/sahilchaddha/Downloads/WhatsApp Image 2025-02-12 at 03.56.20.jpeg",
                       help='Path to the input image')
    args = parser.parse_args()
    
    image = cv2.imread(args.image)
    results = map_pieces_to_positions(image)
    original_with_corners, warped_image, corners, warped_corners, boxes, warped_boxes, piece_positions = results

    # Print coordinates and positions
    print("\nPiece positions:")
    for square_name, box in piece_positions:
        print(f"Piece at {square_name}")

    # Display both images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(cv2.cvtColor(original_with_corners, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image with Detected Corners")
    ax1.axis("off")
    
    ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Warped Image with Corners and Positions")
    ax2.axis("off")
    
    plt.show()
