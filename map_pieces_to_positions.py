import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from shapely.geometry import Polygon
from chesscog.corner_detection.detect_corners import find_corners, resize_image
import threading
import chess

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
    corner_debug_image: Optional[np.ndarray] = None
    piece_detection_image: Optional[np.ndarray] = None

class BoardDetectionException(Exception):
    """Custom exception for board detection errors"""
    pass

class ChessboardDetector:
    def __init__(self, model=None, model_path='runs/detect/train/weights/last.pt', debug_mode=False):
        """
        Initialize detector with either an existing model or a path to model weights
        
        Args:
            model: Existing YOLO model instance (optional)
            model_path: Path to model weights (used only if model is None)
            debug_mode: Enable debug visualization
        """
        from recap import URI, CfgNode as CN
        self.cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
        self.cfg.defrost()
        self.cfg.RANSAC.BEST_SOLUTION_TOLERANCE = 0.1
        self.cfg.RANSAC.OFFSET_TOLERANCE = 0.1
        self.cfg.freeze()
        
        # Use provided model or create new one
        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            
        self.debug_mode = debug_mode

        # Add new attributes for corner caching
        self.cached_corners = None
        self.cached_homography = None
        self.cached_dims = None
        self.corner_quality_score = 0.90
        self.corner_lock = threading.Lock()
        self.force_corner_detection = False

        # Add board state reference
        self.current_board = None

    def _detect_pieces(self, image: np.ndarray):
        """Detect pieces in the image"""
        results = self.model.predict(image)
        return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()

    def _detect_board(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect chessboard corners and get homography matrix"""
        try:
            return find_corners(self.cfg, image)
        except Exception as e:
            raise BoardDetectionException(f"Failed to detect chess board: {str(e)}") from e

    def _create_chess_grid(self, warped_corners: np.ndarray, black_on_left: bool = False):
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
                if black_on_left:
                    # Black's perspective (a1 at bottom right):
                    # - For a cell at grid position (i, j) where i=0 is top row and 7 is bottom,
                    #   we want the row to be counted from the bottom, hence: file letter = chr(97 + (7-i))
                    # - For the column, the rightmost cell (j=7) should be 1, so rank = 8 - j.
                    square_name = f"{chr(97 + (7 - i))}{8 - j}"
                else:
                    # White's perspective (a1 at top left):
                    # - For a cell at grid position (i, j) where i=0 is top row, file letter = chr(97 + i)
                    # - And j=0 (leftmost) gives rank 1, so rank = j + 1.
                    square_name = f"{chr(97 + i)}{j+1}"
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

    def _validate_corners(self, boxes: np.ndarray, corners: np.ndarray, homography: np.ndarray, dims: Tuple) -> float:
        """
        Validate corner detection quality by checking if detected pieces fall within the board
        and are mapped to valid squares. Also validates against current board state if available.
        
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            # Warp piece boxes
            warped_boxes = []
            for box in boxes:
                points = np.array([
                    [box[0], box[1]],  # top-left
                    [box[2], box[1]],  # top-right
                    [box[2], box[3]],  # bottom-right
                    [box[0], box[3]]   # bottom-left
                ])
                points_h = np.hstack([points, np.ones((4, 1))])
                warped_points = homography @ points_h.T
                warped_points = warped_points / warped_points[2]
                warped_points = warped_points[:2].T
                
                # Get bounding box of warped points
                x1, y1 = np.min(warped_points, axis=0)
                x2, y2 = np.max(warped_points, axis=0)
                warped_boxes.append([x1, y1, x2, y2])
            
            # Warp corners
            corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
            warped_corners = homography @ corners_homogeneous.T
            warped_corners = warped_corners / warped_corners[2]
            warped_corners = warped_corners[:2].T

            # Create chess grid
            squares = self._create_chess_grid(warped_corners, black_on_left=True)
            
            # Count pieces that map to valid squares
            valid_mappings = 0
            board_state_matches = 0
            total_pieces = len(warped_boxes)
            
            # Get current board piece positions if available
            board_pieces = {}
            if self.current_board is not None:
                for square in chess.SQUARES:
                    piece = self.current_board.piece_at(square)
                    if piece is not None:
                        board_pieces[chess.square_name(square)] = True

            mapped_squares = set()  # Track mapped squares to avoid double counting
            
            for warped_box in warped_boxes:
                # Use bottom half of warped box for mapping
                mid_y = (warped_box[1] + warped_box[3]) / 2
                box_bottom = np.array([
                    [warped_box[0], mid_y],
                    [warped_box[2], mid_y],
                    [warped_box[2], warped_box[3]],
                    [warped_box[0], warped_box[3]]
                ])
                
                best_square = None
                max_overlap = 0
                
                # Find square with highest overlap
                for square_coords, square_name in squares:
                    overlap = Polygon(box_bottom).intersection(Polygon(square_coords)).area / Polygon(box_bottom).area
                    if overlap > 0.5 and overlap > max_overlap:  # 50% overlap threshold
                        max_overlap = overlap
                        best_square = square_name
                
                if best_square:
                    valid_mappings += 1
                    mapped_squares.add(best_square)
                    
                    # Check if this mapping matches current board state
                    if self.current_board is not None:
                        if best_square in board_pieces:
                            board_state_matches += 1

            # Calculate quality scores
            mapping_score = valid_mappings / total_pieces if total_pieces > 0 else 0.0
            
            # Calculate board state matching score
            if self.current_board is not None:
                expected_pieces = len(board_pieces)
                if expected_pieces > 0:
                    # Consider both accuracy and completeness of detection
                    precision = board_state_matches / total_pieces if total_pieces > 0 else 0.0
                    recall = board_state_matches / expected_pieces
                    board_score = (precision + recall) / 2
                else:
                    board_score = 0.0
            else:
                board_score = 1.0  # Don't penalize if no board state available
            
            # Combine scores (give more weight to board state matching)
            final_score = (mapping_score * 0.4) + (board_score * 0.6)
            
            if self.debug_mode:
                print(f"Corner validation - Mapping score: {mapping_score:.2f}, "
                      f"Board score: {board_score:.2f}, Final score: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            print(f"Corner validation failed: {e}")
            return 0.0

    def process_frame(self, frame: np.ndarray, current_board: Optional[chess.Board] = None, 
                     boxes: Optional[np.ndarray] = None, 
                     confidences: Optional[np.ndarray] = None) -> ChessboardState:
        """
        Process a single frame and return the chess state
        
        Args:
            frame: Input frame
            current_board: Current chess board state (optional)
            boxes: Optional pre-detected piece boxes
            confidences: Optional confidence scores for boxes
        """
        # Update current board reference
        self.current_board = current_board
        
        try:
            frame, scale = resize_image(self.cfg, frame)
            
            # Use provided boxes or detect pieces
            if boxes is None or confidences is None:
                boxes, confidences = self._detect_pieces(frame)        


            corners = None
            homography = None
            dims = None
            
            with self.corner_lock:
                # Detect board and get transformation
                corners, homography, dims = self._detect_board(frame)
                
                # Validate new corner detection
                quality_score = self._validate_corners(boxes, corners, homography, dims)
                if self.force_corner_detection or self.cached_corners is None or quality_score > self.corner_quality_score:
                    self.cached_corners = corners
                    self.cached_homography = homography
                    self.cached_dims = dims
                    self.corner_quality_score = quality_score
                    print(f"Updated corner cache with quality score: {quality_score:.2f}")
                    self.force_corner_detection = False

                # Use cached corners
                corners = self.cached_corners
                homography = self.cached_homography
                dims = self.cached_dims

            # Create corner debug image
            corner_debug_image = frame.copy()
            for i, corner in enumerate(corners):
                cv2.circle(corner_debug_image, 
                          (int(corner[0]), int(corner[1])), 
                          5, (0, 255, 0), -1)
                cv2.putText(corner_debug_image, str(i+1), 
                           (int(corner[0])+10, int(corner[1])+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Create piece detection debug image
            piece_detection_image = frame.copy()
            for box, conf in zip(boxes, confidences):
                cv2.rectangle(piece_detection_image,
                             (int(box[0]), int(box[1])),
                             (int(box[2]), int(box[3])),
                             (0, 0, 255), 2)
                cv2.putText(piece_detection_image,
                           f"{conf:.2f}",
                           (int(box[0]), int(box[1] - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Warp image and corners
            warped_image = cv2.warpPerspective(frame, homography, dims)
            
            # Warp corners
            corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
            warped_corners = homography @ corners_homogeneous.T
            warped_corners = warped_corners / warped_corners[2]
            warped_corners = warped_corners[:2].T
            
            # Warp piece boxes for square mapping only
            warped_boxes = []
            for box in boxes:
                # Convert box corners to points
                points = np.array([
                    [box[0], box[1]],  # top-left
                    [box[2], box[1]],  # top-right
                    [box[2], box[3]],  # bottom-right
                    [box[0], box[3]]   # bottom-left
                ])
                # Add homogeneous coordinate
                points_h = np.hstack([points, np.ones((4, 1))])
                # Transform points
                warped_points = homography @ points_h.T
                warped_points = warped_points / warped_points[2]
                warped_points = warped_points[:2].T
                # Get bounding box of warped points
                x1, y1 = np.min(warped_points, axis=0)
                x2, y2 = np.max(warped_points, axis=0)
                warped_boxes.append([x1, y1, x2, y2])
            warped_boxes = np.array(warped_boxes)
            
            # Determine board orientation using warped boxes
            black_on_left = True
            
            # Create chess grid in warped space
            squares = self._create_chess_grid(warped_corners, black_on_left)
            
            # Map pieces to squares using warped coordinates but keep original boxes
            pieces = []
            debug_image = frame.copy()
            
            # First pass: Calculate overlaps for all pieces
            square_mappings = {}  # Dict to store all pieces mapped to each square
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                warped_box = warped_boxes[i]
                # Use bottom half of warped box
                mid_y = (warped_box[1] + warped_box[3]) / 2
                box_bottom = np.array([
                    [warped_box[0], mid_y],
                    [warped_box[2], mid_y],
                    [warped_box[2], warped_box[3]],
                    [warped_box[0], warped_box[3]]
                ])

                # Find square with highest overlap ratio
                best_square = None
                max_overlap = 0
                
                for square_coords, square_name in squares:
                    # Calculate overlap using original method
                    overlap = Polygon(box_bottom).intersection(Polygon(square_coords)).area / Polygon(box_bottom).area
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_square = square_name
                
                if best_square:
                    if best_square not in square_mappings:
                        square_mappings[best_square] = []
                    square_mappings[best_square].append((box, conf, overlap))

            # Second pass: Select best piece for each square
            for square_name, candidates in square_mappings.items():
                if candidates:
                    # Sort by overlap ratio and take the highest
                    best_candidate = max(candidates, key=lambda x: x[2])
                    box, conf, overlap = best_candidate
                    
                    # Add piece with original box coordinates
                    pieces.append(ChessPiece(square_name, box, conf))
                    
                    # Draw debug visualization
                    cv2.rectangle(debug_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 255, 0), 2)
                    cv2.putText(debug_image,
                               f"{square_name} ({conf:.2f})",
                               (int(box[0]), int(box[1] - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw overlap ratio for debugging
                    if self.debug_mode:
                        cv2.putText(debug_image,
                                   f"o:{overlap:.2f}",
                                   (int(box[0]), int(box[3] + 15)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            return ChessboardState(
                pieces=pieces,
                warped_image=warped_image,
                debug_image=debug_image,
                corner_debug_image=corner_debug_image,
                piece_detection_image=piece_detection_image
            )
        except Exception as e:
            raise BoardDetectionException(f"Failed to process frame: {str(e)}") from e

    def _create_debug_visualization(self, warped_image, corners, chess_squares, pieces, warped_boxes):
        """Create debug visualization image"""
        vis_img = warped_image.copy()
        
        # Draw grid
        for square_coords, _ in chess_squares:
            for i in range(4):
                pt1 = tuple(map(int, square_coords[i]))
                pt2 = tuple(map(int, square_coords[(i + 1) % 4]))
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Create set of mapped boxes for quick lookup
        mapped_boxes = {tuple(piece.box.tolist()) for piece in pieces}
        
        # Draw all warped boxes
        for box in warped_boxes:
            box_tuple = tuple(box.tolist())
            is_mapped = box_tuple in mapped_boxes
            
            # Red for mapped pieces, Yellow for unmapped pieces
            color = (0, 0, 255) if is_mapped else (0, 255, 255)
            
            cv2.rectangle(vis_img, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
            
            # Add square notation for mapped pieces
            if is_mapped:
                piece = next(p for p in pieces if tuple(p.box.tolist()) == box_tuple)
                cv2.putText(vis_img, f"{piece.square} ({piece.confidence:.2f})",
                           (int((box[0] + box[2])/2), int((box[1] + box[3])/2)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return vis_img

    def force_corner_detection_update(self):
        """Force a new corner detection on the next frame"""
        with self.corner_lock:
            self.force_corner_detection = True

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
    
    # Show all debug visualizations
    if args.debug:
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        
        # Original image
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # Piece detection
        if result.piece_detection_image is not None:
            axs[0, 1].imshow(cv2.cvtColor(result.piece_detection_image, cv2.COLOR_BGR2RGB))
            axs[0, 1].set_title('Piece Detection')
            axs[0, 1].axis('off')
        
        # Corner detection
        if result.corner_debug_image is not None:
            axs[0, 2].imshow(cv2.cvtColor(result.corner_debug_image, cv2.COLOR_BGR2RGB))
            axs[0, 2].set_title('Corner Detection')
            axs[0, 2].axis('off')
        
        # Warped image
        if result.warped_image is not None:
            axs[1, 0].imshow(cv2.cvtColor(result.warped_image, cv2.COLOR_BGR2RGB))
            axs[1, 0].set_title('Warped Image')
            axs[1, 0].axis('off')
        
        # Final debug visualization
        if result.debug_image is not None:
            axs[1, 1].imshow(cv2.cvtColor(result.debug_image, cv2.COLOR_BGR2RGB))
            axs[1, 1].set_title('Piece Mapping')
            axs[1, 1].axis('off')
        
        # Clear unused subplot
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
