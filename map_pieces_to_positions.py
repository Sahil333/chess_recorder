import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
from shapely.geometry import Polygon
from chesscog.corner_detection.detect_corners import find_corners

@dataclass
class ChessPiece:
    square: str  # chess notation (e.g., 'e4')
    box: np.ndarray  # [x1, y1, x2, y2]
    confidence: float = 1.0

@dataclass
class ChessboardState:
    pieces: List[ChessPiece]
    warped_image: Optional[np.ndarray] = None
    debug_image: Optional[np.ndarray] = None

class ChessboardDetector:
    def __init__(self, model_path="runs/detect/train/weights/last.pt", debug_mode=False):
        """Initialize the detector with model and configuration"""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.debug_mode = debug_mode
        
        # Cache the configuration
        from recap import URI, CfgNode as CN
        self.cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
        self.cfg.defrost()
        self.cfg.RANSAC.BEST_SOLUTION_TOLERANCE = 0.1
        self.cfg.RANSAC.OFFSET_TOLERANCE = 0.1
        self.cfg.freeze()

    def _detect_pieces(self, image: np.ndarray):
        """Detect pieces in the image"""
        results = self.model.predict(image)
        return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()

    def _detect_board(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect chessboard corners and get homography matrix"""
        return find_corners(self.cfg, image)

    def _create_chess_grid(self, warped_corners: np.ndarray, needs_flip: bool = False):
        """Create 8x8 grid coordinates"""
        x_coords = np.linspace(warped_corners[0][0], warped_corners[1][0], 9)
        y_coords = np.linspace(warped_corners[0][1], warped_corners[3][1], 9)
        
        squares = []
        for i in range(8):
            for j in range(8):
                square = np.array([
                    [x_coords[j], y_coords[i]],
                    [x_coords[j+1], y_coords[i]],
                    [x_coords[j+1], y_coords[i+1]],
                    [x_coords[j], y_coords[i+1]]
                ])
                if needs_flip:
                    square_name = f"{chr(97 + (7-i))}{j+1}"
                else:
                    square_name = f"{chr(97 + i)}{8-j}"
                squares.append((square, square_name))
        return squares

    def _determine_orientation(self, warped_boxes: np.ndarray, warped_image: np.ndarray, warped_corners: np.ndarray) -> bool:
        """Determine board orientation using piece colors"""
        mid_x = warped_corners[0][0] + (warped_corners[1][0] - warped_corners[0][0]) / 2
        left_boxes = warped_boxes[warped_boxes[:, 0] < mid_x]
        right_boxes = warped_boxes[warped_boxes[:, 0] >= mid_x]
        
        def avg_intensity(boxes):
            if len(boxes) == 0:
                return 0
            regions = [warped_image[int(y1):int(y2), int(x1):int(x2)] 
                      for x1, y1, x2, y2 in boxes]
            if len(regions[0].shape) == 3:
                regions = [cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) for r in regions]
            return np.mean([np.mean(r) for r in regions])
        
        left_intensity = avg_intensity(left_boxes)
        right_intensity = avg_intensity(right_boxes)
        return left_intensity < right_intensity  # black_on_left

    def process_frame(self, frame: np.ndarray) -> ChessboardState:
        """Process a single frame and return the chess state"""
        # Detect board and get transformation
        corners, homography = self._detect_board(frame)
        
        # Detect pieces
        boxes, confidences = self._detect_pieces(frame)
        
        # Warp image, corners and boxes
        warped_image = cv2.warpPerspective(frame, homography, frame.shape[1::-1])

        # Warp corners
        # Convert corners to homogeneous coordinates
        corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
        
        # Transform corners using homography matrix
        warped_corners = homography @ corners_homogeneous.T
        # Convert back from homogeneous coordinates
        warped_corners = warped_corners / warped_corners[2]
        warped_corners = warped_corners[:2].T

        warped_boxes = []
        for box in boxes:
            box_corners = np.array([[box[0], box[1]], [box[2], box[1]], 
                                  [box[2], box[3]], [box[0], box[3]]])
            box_corners_h = np.hstack([box_corners, np.ones((4, 1))])
            warped_box_corners = homography @ box_corners_h.T
            warped_box_corners = warped_box_corners / warped_box_corners[2]
            warped_box_corners = warped_box_corners[:2].T
            warped_box = [np.min(warped_box_corners[:, 0]), np.min(warped_box_corners[:, 1]),
                         np.max(warped_box_corners[:, 0]), np.max(warped_box_corners[:, 1])]
            warped_boxes.append(warped_box)
        warped_boxes = np.array(warped_boxes)
        
        # Determine orientation and create grid
        black_on_left = self._determine_orientation(warped_boxes, warped_image, warped_corners)
        chess_squares = self._create_chess_grid(warped_corners, black_on_left)
        
        # Map pieces to squares
        pieces = []
        for box, conf in zip(warped_boxes, confidences):
            # Use bottom half of box
            mid_y = (box[1] + box[3]) / 2
            box_bottom = np.array([[box[0], mid_y], [box[2], mid_y],
                                 [box[2], box[3]], [box[0], box[3]]])
            
            # Find best square
            max_overlap = 0
            best_square = None
            for square_coords, square_name in chess_squares:
                overlap = Polygon(box_bottom).intersection(Polygon(square_coords)).area / Polygon(box_bottom).area
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_square = square_name
            
            if best_square:
                pieces.append(ChessPiece(best_square, box, conf))
        
        # Create debug visualization if needed
        debug_image = None
        if self.debug_mode:
            debug_image = self._create_debug_visualization(
                warped_image, corners, chess_squares, pieces)
        
        return ChessboardState(pieces, warped_image, debug_image)

    def _create_debug_visualization(self, warped_image, corners, chess_squares, pieces):
        """Create debug visualization image"""
        vis_img = warped_image.copy()
        
        # Draw grid
        for square_coords, _ in chess_squares:
            for i in range(4):
                pt1 = tuple(map(int, square_coords[i]))
                pt2 = tuple(map(int, square_coords[(i + 1) % 4]))
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Draw pieces
        for piece in pieces:
            box = piece.box
            cv2.rectangle(vis_img, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (0, 0, 255), 2)
            cv2.putText(vis_img, f"{piece.square} ({piece.confidence:.2f})",
                       (int((box[0] + box[2])/2), int((box[1] + box[3])/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return vis_img

# Example usage
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    np.random.seed(11)
    
    parser = argparse.ArgumentParser(description='Chess board detection and piece mapping')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    args = parser.parse_args()
    
    # Initialize detector
    detector = ChessboardDetector(debug_mode=args.debug)
    
    # Process image
    image = cv2.imread(args.image)
    result = detector.process_frame(image)
    
    # Print results
    print("\nDetected pieces: Total: ", len(result.pieces))
    for piece in result.pieces:
        print(f"Piece at {piece.square} (confidence: {piece.confidence:.2f})")
    
    # Show debug visualization if enabled
    if args.debug and result.debug_image is not None:
        plt.imshow(cv2.cvtColor(result.debug_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
