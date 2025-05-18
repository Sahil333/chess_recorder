import 'dart:typed_data';
import 'package:chess/chess.dart' as chess;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

/// Represents a chess piece detected on the board
class ChessPiece {
  /// Chess notation square (e.g., 'e4')
  final String square;

  /// Bounding box coordinates [x1, y1, x2, y2]
  final Float64List box;

  /// Detection confidence
  final double confidence;

  /// Piece type ('p', 'n', 'b', 'r', 'q', 'k')
  /// Lowercase for black, uppercase for white
  String? pieceType;

  /// Tracking ID (if using tracking)
  int? trackId;

  ChessPiece({
    required this.square,
    required this.box,
    required this.confidence,
    this.pieceType,
    this.trackId,
  });

  /// Create a copy with updated values
  ChessPiece copyWith({
    String? square,
    Float64List? box,
    double? confidence,
    String? pieceType,
    int? trackId,
  }) {
    return ChessPiece(
      square: square ?? this.square,
      box: box ?? this.box,
      confidence: confidence ?? this.confidence,
      pieceType: pieceType ?? this.pieceType,
      trackId: trackId ?? this.trackId,
    );
  }
}

/// Represents the state of a chess board detection
class ChessboardState {
  /// Detected pieces on the board
  final List<ChessPiece> pieces;

  /// Warped (perspective-corrected) image of the board
  final img.Image? warpedImage;

  /// Debug image showing piece mappings
  final img.Image? debugImage;

  /// Debug image showing corner detection
  final img.Image? cornerDebugImage;

  /// Debug image showing piece detection
  final img.Image? pieceDetectionImage;

  /// Quality score of the detection (0-1)
  final double qualityScore;

  ChessboardState({
    required this.pieces,
    this.warpedImage,
    this.debugImage,
    this.cornerDebugImage,
    this.pieceDetectionImage,
    this.qualityScore = 1.0,
  });
}

/// Represents a tracked piece across frames
class TrackedPiece {
  /// Tracking ID assigned by the tracker
  final int trackId;

  /// Piece type ('p', 'n', 'b', 'r', 'q', 'k')
  /// Lowercase for black, uppercase for white
  final String pieceType;

  /// Current square in algebraic notation
  final String currentSquare;

  /// Detection confidence
  final double confidence;

  /// Bounding box coordinates [x1, y1, x2, y2]
  final Float64List box;

  TrackedPiece({
    required this.trackId,
    required this.pieceType,
    required this.currentSquare,
    required this.confidence,
    required this.box,
  });

  /// Create a copy with updated values
  TrackedPiece copyWith({
    int? trackId,
    String? pieceType,
    String? currentSquare,
    double? confidence,
    Float64List? box,
  }) {
    return TrackedPiece(
      trackId: trackId ?? this.trackId,
      pieceType: pieceType ?? this.pieceType,
      currentSquare: currentSquare ?? this.currentSquare,
      confidence: confidence ?? this.confidence,
      box: box ?? this.box,
    );
  }
}

/// Represents a detected move
class DetectedMove {
  /// The algebraic notation of the move (e.g., 'e2e4')
  final String moveNotation;

  /// The chess.Move object
  final chess.Move move;

  /// The confidence of the move detection
  final double confidence;

  /// The frame time when the move was detected
  final double frameTime;

  DetectedMove({
    required this.moveNotation,
    required this.move,
    required this.confidence,
    required this.frameTime,
  });
}

/// Configuration options for the chess recorder
class ChessRecorderConfig {
  /// Debug mode flag
  final bool debugMode;

  /// Confidence threshold for piece detection
  final double pieceConfidenceThreshold;

  /// Minimum number of consistent detections required for a move
  final int minMoveDetections;

  /// Time window for move buffering (in milliseconds)
  final int moveBufferWindowMs;

  /// Paths to TFLite models
  final String pieceDetectorModelPath;
  final String pieceClassifierModelPath;
  final String colorClassifierModelPath;

  /// Initial board position (FEN)
  final String? initialPosition;

  const ChessRecorderConfig({
    this.debugMode = false,
    this.pieceConfidenceThreshold = 0.5,
    this.minMoveDetections = 3,
    this.moveBufferWindowMs = 500,
    this.pieceDetectorModelPath =
        'models/tflite/piece_detector/best_float32.tflite',
    this.pieceClassifierModelPath =
        'models/tflite/piece_classifier/best_float32.tflite',
    this.colorClassifierModelPath =
        'models/tflite/white_black_classifier/white_black_classifier.tflite',
    this.initialPosition,
  });

  /// Create a copy with updated values
  ChessRecorderConfig copyWith({
    bool? debugMode,
    double? pieceConfidenceThreshold,
    int? minMoveDetections,
    int? moveBufferWindowMs,
    String? pieceDetectorModelPath,
    String? pieceClassifierModelPath,
    String? colorClassifierModelPath,
    String? initialPosition,
  }) {
    return ChessRecorderConfig(
      debugMode: debugMode ?? this.debugMode,
      pieceConfidenceThreshold:
          pieceConfidenceThreshold ?? this.pieceConfidenceThreshold,
      minMoveDetections: minMoveDetections ?? this.minMoveDetections,
      moveBufferWindowMs: moveBufferWindowMs ?? this.moveBufferWindowMs,
      pieceDetectorModelPath:
          pieceDetectorModelPath ?? this.pieceDetectorModelPath,
      pieceClassifierModelPath:
          pieceClassifierModelPath ?? this.pieceClassifierModelPath,
      colorClassifierModelPath:
          colorClassifierModelPath ?? this.colorClassifierModelPath,
      initialPosition: initialPosition ?? this.initialPosition,
    );
  }
}
