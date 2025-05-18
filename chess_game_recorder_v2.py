import math
import traceback
import cv2
import chess
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
from map_pieces_to_positions import ChessboardDetector, BoardDetectionException, ChessPiece
from ultralytics import YOLO
import time
from argparse import Namespace
from chesscog.corner_detection.detect_corners import resize_image
from collections import Counter
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
from white_or_black_classifier import SimpleCNN
from PIL import Image
import ffmpeg  # Add this at the top with other imports

@dataclass
class TrackedPiece:
    track_id: int
    piece_type: str  # 'p', 'n', 'b', 'r', 'q', 'k' (lowercase for black, uppercase for white)
    current_square: str
    confidence: float
    box: np.ndarray  # [x1, y1, x2, y2]

class ChessGameRecorderV2:
    def __init__(self, model_path='models/original/piece_detector/best.pt', debug_mode=True):
        # Initialize YOLO model for both detection and tracking
        self.model = YOLO(model_path)

        from recap import URI, CfgNode as CN
        self.cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
        self.cfg.defrost()
        self.cfg.RANSAC.BEST_SOLUTION_TOLERANCE = 0.1
        self.cfg.RANSAC.OFFSET_TOLERANCE = 0.1
        self.cfg.freeze()
        
        
        # Set tracking parameters
        self.tracker_config = {
            'tracker_type': 'bytetrack',  # Use ByteTrack algorithm
            'track_high_thresh': 0.5,     # High confidence threshold
            'track_low_thresh': 0.1,      # Low confidence threshold
            'new_track_thresh': 0.6,      # New track threshold
            'track_buffer': 5,           # Frames to keep track alive
            'match_thresh': 0.3,          # IOU threshold for matching
        }
        
        # Initialize board detector with our model instance
        self.board_detector = ChessboardDetector(model=self.model, debug_mode=debug_mode)
        
        # Initialize chess board
        self.board = chess.Board()
        
        # Store tracked pieces
        self.tracked_pieces: Dict[int, TrackedPiece] = {}
        
        # Debug mode
        self.debug_mode = debug_mode
        
        # Initial position piece mapping
        self.initial_position = {
            # White pieces
            'a1': 'R', 'b1': 'N', 'c1': 'B', 'd1': 'Q', 
            'e1': 'K', 'f1': 'B', 'g1': 'N', 'h1': 'R',
            'a2': 'P', 'b2': 'P', 'c2': 'P', 'd2': 'P',
            'e2': 'P', 'f2': 'P', 'g2': 'P', 'h2': 'P',
            # Black pieces
            'a8': 'r', 'b8': 'n', 'c8': 'b', 'd8': 'q',
            'e8': 'k', 'f8': 'b', 'g8': 'n', 'h8': 'r',
            'a7': 'p', 'b7': 'p', 'c7': 'p', 'd7': 'p',
            'e7': 'p', 'f7': 'p', 'g7': 'p', 'h7': 'p',
        }
        
        # Store previous frame and detection for capture move analysis
        self.prev_frame = None
        self.prev_detection = {}
        
        # Initialize the piece color classifier from the best model produced in training.
        # This classifier distinguishes white (label 0) vs black (label 1) piece crops.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.piece_color_classifier = SimpleCNN().to(self.device)
        try:
            self.piece_color_classifier.load_state_dict(torch.load("models/original/white_black_classifier/best_white_black_classifier.pth", map_location=self.device))
            self.piece_color_classifier.eval()
            print("Loaded piece color classifier successfully.")
        except Exception as e:
            print(f"Failed to load piece color classifier: {e}")
            self.piece_color_classifier = None

        # Define transform identical to training (resize to 75x75, to tensor, and normalize)
        self.classifier_transform = transforms.Compose([
            transforms.Resize((75, 75)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Add new attributes for move buffering
        self.move_buffer = []  # List of (frame_time, move) tuples
        self.move_buffer_window = 0.5  # 500ms window
        self.min_move_detections = 3  # Minimum number of consistent detections required
        self.last_committed_move_time = None
        self.last_committed_move = None
        self.fps = None  # Will be set when processing video
        
        # Add classification model for piece type identification
        self.classification_model = YOLO('models/original/piece_classifier/best.pt')
    
    def process_move_buffer(self, frame_time: float) -> Optional[chess.Move]:
        """
        Process the move buffer to find the most consistent move detection.
        Returns the most frequent legal move if it meets the threshold criteria.
        
        Args:
            frame_time: Time in seconds from the start of the video
        """
        # Remove old moves outside the time window
        cutoff_time = frame_time - self.move_buffer_window
        self.move_buffer = [(t, m) for t, m in self.move_buffer if t > cutoff_time]
        
        if not self.move_buffer:
            return None

        # Count move occurrences
        move_counts = Counter(str(move) for _, move in self.move_buffer)
        
        # Find the most common move
        most_common_move, count = move_counts.most_common(1)[0]
        
        # Check if we have enough consistent detections
        if count >= self.min_move_detections:
            # Clear the buffer after committing a move
            self.move_buffer.clear()
            return chess.Move.from_uci(most_common_move)
            
        return None

    def detect_moves(self, detected_pieces: List, board_state=None, current_frame=None, frame_time: float = 0) -> Optional[chess.Move]:
        """
        Detect chess moves from detected pieces and add to move buffer.
        Returns a move only when there are enough consistent detections.
        
        Args:
            frame_time: Time in seconds from the start of the video
        """
        if not detected_pieces:
            return None
        
        # Skip move detection if we recently committed a move
        if (self.last_committed_move_time and 
            frame_time - self.last_committed_move_time < self.move_buffer_window):
            return None

        # Get current board squares and detected squares
        detected_squares = {piece.square for piece in detected_pieces}
        current_squares = {chess.square_name(sq) for sq in self.board.piece_map().keys()}
        
        # Find differences
        new_squares = detected_squares - current_squares
        old_squares = current_squares - detected_squares
        
        detected_move = None
        
        # Simple move analysis
        if len(new_squares) == 1 and len(old_squares) == 1:
            from_square = old_squares.pop()
            to_square = new_squares.pop()
            move = chess.Move.from_uci(f"{from_square}{to_square}")

            if ((chess.square_rank(chess.parse_square(to_square)) == 7 and 
                 self.board.piece_at(chess.parse_square(from_square)).symbol().lower() == 'p' and 
                 self.board.turn) or 
                (chess.square_rank(chess.parse_square(to_square)) == 0 and 
                 self.board.piece_at(chess.parse_square(from_square)).symbol().lower() == 'p' and 
                 not self.board.turn)):
                
                # Find the piece at the destination square in detected pieces
                promoted_piece = next((p for p in detected_pieces if p.square == to_square), None)
                if promoted_piece:
                    # Extract the piece image from the frame
                    box = promoted_piece.box.astype(int)
                    piece_img = current_frame[box[1]:box[3], box[0]:box[2]]
                    
                    # Use classification model to predict piece type
                    results = self.classification_model(piece_img)
                    class_id = results[0].probs.top1
                    confidence = results[0].probs.top1conf
                    
                    # Map class index to piece type (adjust these indices based on your model's classes)
                    piece_types = ['b', 'k', 'p', 'q', 'r']  # Possible promotion pieces
                    if class_id < len(piece_types) and confidence > 0.5 and class_id != 2:
                        promotion_piece = piece_types[class_id]
                        # Create promotion move
                        move = chess.Move.from_uci(f"{from_square}{to_square}{promotion_piece}")

            # check if the move is a promotion move.
            if self.board.is_legal(move):
                detected_move = move
            else:
                detected_move = None
        # Capture move analysis
        elif (len(new_squares) == 0 and len(old_squares) == 1 and 
              board_state is not None and current_frame is not None and self.prev_detection):
            from_square = old_squares.pop()
            moving_piece = self.board.piece_at(chess.parse_square(from_square))
            if moving_piece is None:
                return None

            # Consider only capturing moves
            candidate_moves = [m for m in self.board.legal_moves if m.from_square == chess.parse_square(from_square) and self.board.is_capture(m)]
            best_candidate = None
            for m in candidate_moves:
                dest_square = chess.square_name(m.to_square)
                if dest_square in self.prev_detection:
                    prev_piece = self.prev_detection[dest_square]
                    # Find current detection for the destination square (capturing piece)
                    current_piece = next((p for p in detected_pieces if p.square == dest_square), None)
                    if current_piece:
                        # Extract regions of interest for the pieces from previous and current frames
                        prev_roi = self.prev_frame[math.floor(prev_piece.box[1]):math.floor(prev_piece.box[3]),
                                                     math.floor(prev_piece.box[0]):math.floor(prev_piece.box[2])]
                        current_roi = current_frame[math.floor(current_piece.box[1]):math.floor(current_piece.box[3]),
                                                     math.floor(current_piece.box[0]):math.floor(current_piece.box[2])]

                        # Use the classifier to predict the color of the current ROI.
                        pred_current = self.classify_piece_color(current_roi) if self.piece_color_classifier is not None else None
                        if pred_current is None:
                            continue

                        # Determine the moving piece's color: 0 for white (uppercase symbol), 1 for black.
                        moving_color = 0 if moving_piece.symbol().isupper() else 1
                        # For a valid capture, the piece at the destination should be of the opposite color.
                        if pred_current != moving_color:
                            continue
                        
                        # Check if this is a promotion capture move
                        if ((chess.square_rank(m.to_square) == 7 and 
                             moving_piece.symbol().lower() == 'p' and 
                             self.board.turn) or 
                            (chess.square_rank(m.to_square) == 0 and 
                             moving_piece.symbol().lower() == 'p' and 
                             not self.board.turn)):
                            
                            # Extract the piece image from the frame for classification
                            box = current_piece.box.astype(int)
                            piece_img = current_frame[box[1]:box[3], box[0]:box[2]]
                            
                            # Use classification model to predict piece type
                            results = self.classification_model(piece_img)
                            class_id = results[0].probs.top1
                            confidence = results[0].probs.top1conf
                            
                            # Map class index to piece type
                            piece_types = ['b', 'k', 'p', 'q', 'r']  # Possible promotion pieces
                            if class_id < len(piece_types) and confidence > 0.5 and class_id != 2:
                                promotion_piece = piece_types[class_id]
                                # Create promotion capture move
                                best_candidate = chess.Move.from_uci(f"{from_square}{dest_square}{promotion_piece}")
                                break
                        else:
                            best_candidate = m

            if best_candidate and self.board.is_legal(best_candidate):
                detected_move = best_candidate
        # En passant move analysis
        elif len(old_squares) == 2 and len(new_squares) == 1:
            to_square = new_squares.pop()  # The destination square
            old_squares_list = sorted(list(old_squares))
            
            # Find which old square has our pawn (the capturing piece)
            capturing_square = None
            captured_square = None
            for square in old_squares_list:
                piece = self.board.piece_at(chess.parse_square(square))
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == self.board.turn:
                        capturing_square = square
                    else:  # opponent's pawn
                        captured_square = square
            
            # Only proceed if we found both pawns
            if capturing_square and captured_square:
                # Create the potential en passant move
                move = chess.Move.from_uci(f"{capturing_square}{to_square}")
                if self.board.is_legal(move):
                    detected_move = move
        # Castling move analysis
        elif len(new_squares) == 2 and len(old_squares) == 2:
            new_squares_list = sorted(list(new_squares))
            old_squares_list = sorted(list(old_squares))
            
            # Define castling patterns for both colors
            castling_patterns = {
                # White castling patterns
                'white_kingside': {
                    'old': ['e1', 'h1'],  # king and rook original squares
                    'new': ['f1', 'g1']   # king and rook final squares
                },
                'white_queenside': {
                    'old': ['a1', 'e1'],
                    'new': ['c1', 'd1']
                },
                # Black castling patterns
                'black_kingside': {
                    'old': ['e8', 'h8'],
                    'new': ['f8', 'g8']
                },
                'black_queenside': {
                    'old': ['a8', 'e8'],
                    'new': ['c8', 'd8']
                }
            }
            
            # Check if the squares match any castling pattern based on the current turn
            if self.board.turn:  # White's turn
                if (old_squares_list == castling_patterns['white_kingside']['old'] and 
                    new_squares_list == castling_patterns['white_kingside']['new']):
                    detected_move = chess.Move.from_uci('e1g1')  # White kingside castle
                elif (old_squares_list == castling_patterns['white_queenside']['old'] and 
                      new_squares_list == castling_patterns['white_queenside']['new']):
                    detected_move = chess.Move.from_uci('e1c1')  # White queenside castle
            else:  # Black's turn
                if (old_squares_list == castling_patterns['black_kingside']['old'] and 
                    new_squares_list == castling_patterns['black_kingside']['new']):
                    detected_move = chess.Move.from_uci('e8g8')  # Black kingside castle
                elif (old_squares_list == castling_patterns['black_queenside']['old'] and 
                      new_squares_list == castling_patterns['black_queenside']['new']):
                    detected_move = chess.Move.from_uci('e8c8')  # Black queenside castle
            
            # Verify if the detected castling move is legal
            if detected_move and self.board.is_legal(detected_move):
                pass  # Continue with move buffer processing
            else:
                detected_move = None

        # If we detected a move, add it to the buffer
        if detected_move:
            self.move_buffer.append((frame_time, detected_move))
            
            # Try to process the buffer
            committed_move = self.process_move_buffer(frame_time)
            if committed_move:
                self.last_committed_move_time = frame_time
                self.last_committed_move = committed_move
                return committed_move
        
        return None

    def visualize_tracking(self, frame: np.ndarray, pieces: List) -> np.ndarray:
        """Create visualization of tracked or detected pieces with improved visuals"""
        vis_img = frame.copy()
        
        # Color mapping for piece types
        color_map = {
            'P': (0, 255, 0),    # White pawns: green
            'N': (255, 0, 0),    # White knights: red
            'B': (0, 0, 255),    # White bishops: blue
            'R': (255, 255, 0),  # White rooks: cyan
            'Q': (255, 0, 255),  # White queen: magenta
            'K': (128, 128, 0),  # White king: olive
            'p': (0, 128, 0),    # Black pawns: dark green
            'n': (128, 0, 0),    # Black knights: dark red
            'b': (0, 0, 128),    # Black bishops: dark blue
            'r': (128, 128, 0),    # Black rooks: dark cyan
            'q': (128, 0, 128),    # Black queen: dark magenta
            'k': (64, 64, 0),    # Black king: dark olive
        }
        
        for piece in pieces:
            box = piece.box.astype(int)
            color = color_map.get(piece.piece_type, (0, 255, 0))  # Default to green
            
            # Draw the bounding box with piece-specific color
            cv2.rectangle(vis_img, 
                         (box[0], box[1]), (box[2], box[3]),
                         color, 2)
            
            # Use 'current_square' if available; otherwise, use detected 'square'
            label_square = getattr(piece, 'current_square', piece.square)
            label_id = getattr(piece, 'track_id', 'N/A')
            label = f"{piece.piece_type}:{label_id} ({label_square})"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img,
                         (box[0], box[1] - text_h - 4),
                         (box[0] + text_w, box[1]),
                         color, -1)
            
            cv2.putText(vis_img, label,
                        (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw confidence score
            conf_label = f"{piece.confidence:.2f}"
            cv2.putText(vis_img, conf_label,
                        (box[0], box[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_img

    def check_rotation(self, path_video_file):
        """
        Check if the video requires rotation based on metadata.
        Returns the OpenCV rotation code if rotation is needed, None otherwise.
        """
        try:
            # This returns meta-data of the video file in form of a dictionary
            meta_dict = ffmpeg.probe(path_video_file)
            
            # Get rotation info from the video metadata if it exists
            # Handle cases where the tags might not exist in the metadata
            rotate_tag = meta_dict.get('streams', [{}])[0].get('side_data_list', {})[0].get('rotation', None)
            
            if rotate_tag is None:
                return None
            
            rotate_value = int(rotate_tag)
            rotateCode = None
            
            if rotate_value == 90:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
            elif rotate_value == 180:
                rotateCode = cv2.ROTATE_180
            elif rotate_value == 270:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif rotate_value == -90:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif rotate_value == -180:
                rotateCode = cv2.ROTATE_180
            elif rotate_value == -270:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
            
            return rotateCode
        except Exception as e:
            print(f"Warning: Could not check video rotation: {e}")
            return None

    def correct_rotation(self, frame, rotateCode):
        """Apply rotation correction to a frame"""
        return cv2.rotate(frame, rotateCode)

    def process_video(self, source: Union[str, int], frame_interval: float = 1.0, start_frame: int = 0):
        """Process chess game video"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        current_interval = frame_interval
        frames_to_skip = int(current_interval * self.fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if video needs rotation (only for file sources, not camera)
        rotateCode = None
        if isinstance(source, str):
            rotateCode = self.check_rotation(source)
            if rotateCode is not None:
                print(f"Video rotation detected. Will apply rotation correction.")
        
        # Set starting frame if provided
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
        else:
            frame_count = 0
        
        # Set up visualization
        plt.ion()
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        
        print("\nDebug Controls:")
        print("'q' - Quit")
        print("'s' - Save current frame")
        if self.debug_mode:
            print("Enter - Next frame")
            print("'d' - Increase frame interval")
            print("'a' - Decrease frame interval")
        
        print(f"\nVideo Info:")
        print(f"FPS: {self.fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Processing every {frames_to_skip} frames")
        
        try:
            print(f"\nInitial position:")
            print(self.board)
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Correct frame rotation if needed
                if rotateCode is not None:
                    frame = self.correct_rotation(frame, rotateCode)
                
                # Resize the frame (after rotation)
                frame, scale = resize_image(self.cfg, frame)
                
                frame_count += 1
                if (frame_count - 1) % frames_to_skip != 0:
                    continue
                
                # Calculate frame time in seconds
                frame_time = frame_count / self.fps
                
                print(f"\nProcessing frame {frame_count}/{total_frames} (time: {frame_time:.2f}s)")
                
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Clear previous plots
                for ax in axs.flat:
                    ax.clear()
                
                # Show input frame
                axs[0, 0].imshow(frame_rgb)
                axs[0, 0].set_title('Input Frame')
                axs[0, 0].axis('off')
                
                try:
                    # Process frame and get all debug images
                    board_state = self.board_detector.process_frame(frame, current_board=self.board)
                    
                    if board_state is not None:
                        # Add quality score to debug visualization
                        if self.debug_mode:
                            quality_text = f"Corner Quality: {self.board_detector.corner_quality_score:.2f}"
                            cv2.putText(board_state.corner_debug_image,
                                      quality_text,
                                      (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1,
                                      (0, 255, 0),
                                      2)

                        # Show corner detection debug image
                        if hasattr(board_state, 'corner_debug_image') and board_state.corner_debug_image is not None:
                            corner_rgb = cv2.cvtColor(board_state.corner_debug_image, cv2.COLOR_BGR2RGB)
                            axs[0, 1].imshow(corner_rgb)
                            axs[0, 1].set_title('Corner Detection')
                            axs[0, 1].axis('off')
                        
                        # Show warped board
                        if board_state.warped_image is not None:
                            warped_rgb = cv2.cvtColor(board_state.warped_image, cv2.COLOR_BGR2RGB)
                            axs[0, 2].imshow(warped_rgb)
                            axs[0, 2].set_title('Warped Board')
                            axs[0, 2].axis('off')
                        
                        # Show piece detection results
                        if board_state.piece_detection_image is not None:
                            piece_rgb = cv2.cvtColor(board_state.piece_detection_image, cv2.COLOR_BGR2RGB)
                            axs[1, 0].imshow(piece_rgb)
                            axs[1, 0].set_title('Piece Detection')
                            axs[1, 0].axis('off')
                        
                        # Show piece mapping (debug image)
                        if board_state.debug_image is not None:
                            debug_rgb = cv2.cvtColor(board_state.debug_image, cv2.COLOR_BGR2RGB)
                            axs[1, 1].imshow(debug_rgb)
                            axs[1, 1].set_title('Piece Mapping')
                            axs[1, 1].axis('off')
                        
                        # ------ NEW: Use detected pieces directly instead of tracking ------
                        detected_pieces = board_state.pieces
                        # Assign a temporary track id for visualization purposes
                        for idx, piece in enumerate(detected_pieces):
                            piece.track_id = idx  # temporary id for visualization
                        
                        # Update piece type based on the current board (using standard chess notation)
                        for piece in detected_pieces:
                            try:
                                board_piece = self.board.piece_at(chess.parse_square(piece.square))
                                piece.piece_type = board_piece.symbol() if board_piece is not None else ''
                            except Exception:
                                piece.piece_type = ''
                        
                        # Use our visualization function to draw these detected pieces on the frame
                        tracking_vis = self.visualize_tracking(frame, detected_pieces)
                        axs[1, 2].imshow(cv2.cvtColor(tracking_vis, cv2.COLOR_BGR2RGB))
                        axs[1, 2].set_title('Piece Detection Visualization')
                        axs[1, 2].axis('off')
                        
                        # ------ NEW: Detect moves with buffering ------
                        move = self.detect_moves(detected_pieces, board_state, frame, frame_time)
                        if move:
                            self.board.push(move)
                            print(f"\nCommitted move: {move} (detected consistently over {self.move_buffer_window}s)")
                            print("\nCurrent position:")
                            print(self.board)
                            
                        # Update legal board state snapshot for subsequent capture move analysis
                        self.prev_frame = frame.copy()
                        self.prev_detection = {piece.square: piece for piece in detected_pieces}

                        # Show move buffer info in debug visualization
                        if self.debug_mode:
                            buffer_info = (f"Move buffer size: {len(self.move_buffer)}\n"
                                         f"Last committed move: {self.last_committed_move}\n"
                                         f"Video time: {frame_time:.2f}s\n"
                                         f"Time since last move: {frame_time - self.last_committed_move_time:.2f}s" 
                                         if self.last_committed_move_time else "No moves yet")
                            axs[2, 2].text(0.5, 0.5, buffer_info,
                                         family='monospace', size=10,
                                         ha='center', va='center')
                            axs[2, 2].set_title('Move Buffer Info')
                            axs[2, 2].axis('off')
                        
                        # Show current board state
                        axs[2, 0].text(0.5, 0.5, str(self.board), 
                                    family='monospace', size=10,
                                    ha='center', va='center')
                        axs[2, 0].set_title('Current Board State')
                        axs[2, 0].axis('off')
                        
                        # Show move information
                        move_info = (f"Last Move: {move if move else 'None'}\n"
                                   f"Side to move: {'White' if self.board.turn else 'Black'}\n"
                                   f"Move number: {self.board.fullmove_number}")
                        axs[2, 1].text(0.5, 0.5, move_info,
                                    family='monospace', size=10,
                                    ha='center', va='center')
                        axs[2, 1].set_title('Game Info')
                        axs[2, 1].axis('off')
                    
                    # Clear unused subplot
                    axs[2, 2].axis('off')
                    
                except Exception as e:
                    # print stack trace
                    print(traceback.format_exc())

                    print(f"Board detection failed: {e}")
                    axs[1, 1].text(0.5, 0.5, f"Error:\n{str(e)}", 
                                 ha='center', va='center', wrap=True, color='red')
                    axs[1, 1].axis('off')
                
                # Update the display
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)
                
                # In debug mode, wait for user input
                if self.debug_mode:
                    while True and frame_count > 125000000:
                        command = input("Command (Enter/s/q/d for increase, a for decrease): ").lower().strip()
                        if command == 'q':  # Quit
                            return
                        elif command == 's':  # Save frame
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"debug_frame_{timestamp}_{frame_count}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"\nSaved frame to: {filename}")
                        elif command == 'd':  # Increase interval
                            current_interval += 0.5
                            frames_to_skip = int(current_interval * self.fps)
                            print(f"\nIncreased frame interval to {current_interval} sec, frames to skip: {frames_to_skip}")
                        elif command == 'a':  # Decrease interval
                            current_interval = max(0.1, current_interval - 0.5)
                            frames_to_skip = int(current_interval * self.fps)
                            print(f"\nDecreased frame interval to {current_interval} sec, frames to skip: {frames_to_skip}")
                        elif command == '':  # Enter to continue
                            break
                else:
                    # Non-debug mode: quick check for quit
                    if input("Press 'q' to quit or Enter to continue: ").lower().strip() == 'q':
                        return
        
        finally:
            cap.release()
            plt.close('all')

    def count_occupied_squares(self):
        """
        Returns the number of occupied squares on the chess board.
        """
        return len(self.board.piece_map())

    def classify_piece_color(self, roi: np.ndarray) -> Optional[int]:
        """
        Classify the piece color from a region of interest (ROI).
        Returns:
             0 for white, 1 for black, or None if classification fails.
        """
        try:
            # Convert the image from OpenCV BGR to PIL RGB
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            input_tensor = self.classifier_transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.piece_color_classifier(input_tensor)
            prediction = output.argmax(dim=1).item()
            return prediction
        except Exception as e:
            print(f"Error in classify_piece_color: {e}")
            return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chess game recording with piece tracking')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file or camera index')
    parser.add_argument('--interval', type=float, default=1.0, help='Frame processing interval in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    parser.add_argument('--start-frame', type=int, default=0, help='Frame number to start processing from')
    parser.add_argument('--position', type=str, default=None, help='FEN string representing the starting chess position')
    args = parser.parse_args()
    np.random.seed(42)
    
    # Convert video argument to int if it's a number (camera index)
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Initialize recorder
    recorder = ChessGameRecorderV2(debug_mode=args.debug)
    
    # Set starting chess position if provided; if not, standard starting position is used.
    if args.position:
        try:
            recorder.board = chess.Board(fen=args.position)
            print(f"Using custom starting position: {args.position}")
        except Exception as e:
            print(f"Invalid FEN provided for start position: {args.position}. Using standard starting position.")
    
    try:
        # Process video while starting at the specified frame
        recorder.process_video(video_source, args.interval, start_frame=args.start_frame)
    except Exception as e:
        print(f"Error processing video: {e}") 