/// Chess Recorder module for Flutter
///
/// This module provides functionality to detect and record chess games from a camera feed
///
/// Main components:
/// - ChessRecorder: Main class for processing frames and detecting chess moves
/// - ChessboardDetector: Detects chessboard and maps pieces to positions
/// - CornerDetector: Detects corners of a chessboard
/// - ChessboardState: Represents the state of a detected chessboard
/// - ChessPiece: Represents a detected chess piece
/// - TrackedPiece: Represents a tracked chess piece
/// - ChessRecorderView: Flutter widget for displaying the chess recorder
library;

export 'chess_recorder.dart';
export 'chess_models.dart';
export 'chessboard_detector.dart';
export 'corner_detector.dart';
export 'chess_recorder_view.dart';
