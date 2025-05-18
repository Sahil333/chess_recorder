# Chess game recorder
# Assumptions:
# 1. Video starts from an initial position of chess
# 2. Camera is static throughout the video. Some movement is allowed.
# Strategy
# 1. Take a stream of images from the camera
# 2. Try to detect the chess board in the initial frames and detect the pieces and mark the piece type as well based on initial positions
# 3. From there on, track the movement of the pieces using object tracking and we will develop a model that will take us from one position to another legally 
# 4. One model that we can try is take a frame do some processing on it and if we can find a piece movement that is legal from last legal position, then we can move the piece
# 5. Detecting the piece movement can be done by using the ChessBoardDetector to detect the board and then comparing it with the piece positions from previous legal position to detect the movement

import cv2
import chess
import numpy as np
from map_pieces_to_positions import ChessboardDetector, BoardDetectionException
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
import time
import argparse

class ChessGameRecorder:
    def __init__(self, model_path='runs/detect/train/weights/last.pt', debug_mode=True):
        # Initialize board detector
        self.board_detector = ChessboardDetector(model_path=model_path, debug_mode=debug_mode)
        
        # Initialize chess board with starting position
        self.board = chess.Board()
        
        # Store the previous position state
        self.previous_pieces = None
        
        # Debug mode for visualization
        self.debug_mode = debug_mode
    
    def detect_move(self, current_pieces, previous_pieces):
        """
        Detect the move made by comparing current and previous positions
        Returns a chess.Move object if a valid move is found
        """
        if not previous_pieces:
            return None
            
        # Convert piece lists to sets of squares
        current_squares = {piece.square for piece in current_pieces}
        previous_squares = {piece.square for piece in previous_pieces}
        
        # Find disappeared and appeared squares
        disappeared = previous_squares - current_squares
        appeared = current_squares - previous_squares
        
        if len(disappeared) == 1 and len(appeared) == 1:
            # Simple move: one piece moved from square A to square B
            from_square = chess.parse_square(list(disappeared)[0])
            to_square = chess.parse_square(list(appeared)[0])
            move = chess.Move(from_square, to_square)
            
            # Verify if move is legal
            if move in self.board.legal_moves:
                return move
                
        # TODO: Handle special cases like castling, en passant, and promotion
        return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Process a single frame and update the chess position"""
        if frame is None:
            raise ValueError("Invalid frame")
            
        try:
            # Detect board and pieces
            board_state = self.board_detector.process_frame(frame)
            
            # Detect and validate move
            move = self.detect_move(board_state.pieces, self.previous_pieces)
            
            if move:
                # Make the move on our board
                self.board.push(move)
                print(f"Detected move: {move}")
            
            # Update previous state
            self.previous_pieces = board_state.pieces
            
            # Return current FEN and debug image
            return self.board.fen(), board_state.debug_image
            
        except BoardDetectionException as e:
            print(f"Board detection failed: {e}")
            return None, None

    def process_video(self, source: Union[str, int], frame_interval: float = 1.0):
        """
        Process video from file or camera stream
        
        Args:
            source: Video file path (str) or camera index (int)
            frame_interval: Number of frames to skip between processing
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(frame_interval * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # Set up matplotlib for visualization
        plt.ion()
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        
        print("\nDebug Controls:")
        print("'q' - Quit")
        print("'s' - Save current frame")
        if self.debug_mode:
            print("Enter - Next frame")
        
        print(f"\nVideo Info:")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Processing every {frames_to_skip} frames")
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process only every nth frame
                if (frame_count - 1) % frames_to_skip != 0:
                    continue
                
                print(f"\nProcessing frame {frame_count}/{total_frames}")
                
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
                    board_state = self.board_detector.process_frame(frame)
                    fen = None
                    
                    if board_state is not None:
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
                        if board_state.debug_image is not None:
                            piece_rgb = cv2.cvtColor(board_state.debug_image, cv2.COLOR_BGR2RGB)
                            axs[1, 0].imshow(piece_rgb)
                            axs[1, 0].set_title('Piece Detection')
                            axs[1, 0].axis('off')
                        
                        # Detect and validate move
                        move = self.detect_move(board_state.pieces, self.previous_pieces)
                        if move:
                            self.board.push(move)
                            print(f"\nDetected move: {move}")
                            # Update previous state
                            self.previous_pieces = board_state.pieces
                        
                        fen = self.board.fen()
                        
                        if fen is not None:
                            print("\nCurrent position (FEN):", fen)
                            print("\nBoard visualization:")
                            print(self.board)  # This will print ASCII board
                            print("\nSide to move:", "White" if self.board.turn else "Black")
                            print("Move number:", self.board.fullmove_number)
                    
                    # Show current board state
                    axs[1, 1].text(0.5, 0.5, f"FEN:\n{fen if fen else 'No position'}", 
                                 ha='center', va='center', wrap=True)
                    axs[1, 1].axis('off')
                    
                except BoardDetectionException as e:
                    print(f"Board detection failed: {e}")
                    axs[1, 1].text(0.5, 0.5, f"Error:\n{str(e)}", 
                                 ha='center', va='center', wrap=True, color='red')
                    axs[1, 1].axis('off')
                
                # Show live feed
                axs[1, 2].imshow(frame_rgb)
                axs[1, 2].set_title('Live Feed')
                axs[1, 2].axis('off')
                
                # Update the display
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)  # Small pause to update the plot
                
                # In debug mode, wait for user input
                if self.debug_mode:
                    while True:
                        command = input("Command (Enter/s/q): ").lower().strip()
                        if command == 'q':  # Quit
                            return
                        elif command == 's':  # Save frame
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"debug_frame_{timestamp}_{frame_count}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"\nSaved frame to: {filename}")
                        elif command == '':  # Enter to continue
                            break
                else:
                    # Non-debug mode: quick check for quit
                    if input("Press 'q' to quit or Enter to continue: ").lower().strip() == 'q':
                        return
        
        finally:
            cap.release()
            plt.close('all')

if __name__ == "__main__":
    # np.random.seed(11)
    
    parser = argparse.ArgumentParser(description='Chess game recording from video')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file or camera index')
    parser.add_argument('--interval', type=float, default=1.0, help='Frame processing interval in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    args = parser.parse_args()
    
    # Convert video argument to int if it's a number (camera index)
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Initialize recorder
    recorder = ChessGameRecorder(debug_mode=args.debug)
    
    try:
        # Process video
        recorder.process_video(video_source, args.interval)
    except Exception as e:
        print(f"Error processing video: {e}")


