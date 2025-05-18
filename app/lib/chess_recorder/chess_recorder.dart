import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:chess/chess.dart' as chess;
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:logger/logger.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:tuple/tuple.dart';
import 'package:collection/collection.dart';

import 'chess_models.dart';
import 'corner_detector.dart';
import 'chessboard_detector.dart';

enum RecorderState {
  initialized,
  findingBoard,
  boardFound,
  gameInProgress,
  gamePaused,
  gameComplete,
  error,
}

/// Class that implements chess game recording functionality
class ChessRecorder {
  final Logger logger = Logger();
  final ChessRecorderConfig config;

  // ML models
  late final Interpreter? pieceDetectorModel;
  late final Interpreter? pieceClassifierModel;
  late final Interpreter? colorClassifierModel;

  // Chess state
  late chess.Chess board;

  // Board detector and configuration
  late ChessboardDetector boardDetector;

  // Track pieces
  final Map<int, TrackedPiece> trackedPieces = {};

  // Store previous frame and detection for capture move analysis
  img.Image? prevFrame;
  Map<String, ChessPiece> prevDetection = {};

  // Move buffering
  final List<Tuple2<double, chess.Move>> moveBuffer = [];
  double? lastCommittedMoveTime;
  chess.Move? lastCommittedMove;

  // Camera properties
  double? fps;

  // Current state
  RecorderState _state = RecorderState.initialized;
  RecorderState get state => _state;

  // Callbacks
  final void Function(RecorderState)? onStateChanged;
  final void Function(ChessboardState)? onBoardDetected;
  final void Function(chess.Move)? onMoveDetected;
  final void Function(String)? onError;

  // Initial position mapping
  final Map<String, String> initialPosition = {
    // White pieces
    'a1': 'R', 'b1': 'N', 'c1': 'B', 'd1': 'Q',
    'e1': 'K', 'f1': 'B', 'g1': 'N', 'h1': 'R',
    'a2': 'P', 'b2': 'P', 'c2': 'P', 'd2': 'P',
    'e2': 'P', 'f2': 'P', 'g2': 'P', 'h2': 'P',
    // Black pieces
    'a8': 'r', 'b8': 'n', 'c8': 'b', 'd8': 'q',
    'e8': 'k', 'f8': 'b', 'g8': 'n', 'h8': 'r',
    'a7': 'p', 'b7': 'p', 'c7': 'p', 'd7': 'p',
    'e7': 'p', 'f7': 'p', 'g7': 'p', 'h7': 'p',
  };

  /// Create a new chess recorder
  ChessRecorder({
    ChessRecorderConfig? config,
    this.onStateChanged,
    this.onBoardDetected,
    this.onMoveDetected,
    this.onError,
    this.fps,
  }) : config = config ?? ChessRecorderConfig() {
    // Initialize chess board
    board = chess.Chess();

    if (this.config.initialPosition != null) {
      try {
        board.load(this.config.initialPosition!);
        logger.i(
          "Using custom starting position: ${this.config.initialPosition}",
        );
      } catch (e) {
        logger.e(
          "Invalid FEN provided for start position. Using standard starting position.",
        );
      }
    }

    _state = RecorderState.initialized;
    if (onStateChanged != null) onStateChanged!(_state);
  }

  /// Initialize the ML models
  Future<void> initialize() async {
    try {
      // Load the piece detector model
      pieceDetectorModel = await _loadModel(config.pieceDetectorModelPath);
      logger.i("Piece detector model loaded");

      // Load the piece classifier model
      pieceClassifierModel = await _loadModel(config.pieceClassifierModelPath);
      logger.i("Piece classifier model loaded");

      // Load the color classifier model
      colorClassifierModel = await _loadModel(config.colorClassifierModelPath);
      logger.i("Color classifier model loaded");

      // Initialize the board detector
      boardDetector = ChessboardDetector(
        pieceDetectorModel: pieceDetectorModel,
        debugMode: config.debugMode,
      );

      _state = RecorderState.findingBoard;
      if (onStateChanged != null) onStateChanged!(_state);

      logger.i("Chess recorder initialized");
    } catch (e) {
      logger.e("Failed to initialize chess recorder: $e");
      _state = RecorderState.error;
      if (onStateChanged != null) onStateChanged!(_state);
      if (onError != null) onError!("Failed to initialize: $e");
    }
  }

  /// Load a TFLite model from assets
  Future<Interpreter?> _loadModel(String modelPath) async {
    try {
      final interpreter = await Interpreter.fromAsset(modelPath);
      return interpreter;
    } catch (e) {
      logger.e("Error loading model $modelPath: $e");
      return null;
    }
  }

  /// Process a frame from the camera
  Future<ChessboardState?> processFrame(
    cv.Mat frame,
    double frameTime,
  ) async {
    try {
      if (_state == RecorderState.error) {
        return null;
      }

      // Process the frame to detect the board and pieces
      final boardState = await boardDetector.processFrame(
        frame,
        currentBoard: board,
      );

      // Update state based on detection
      if (_state == RecorderState.findingBoard) {
        if (boardState.pieces.length >= 20) {
          // Most chess pieces detected
          _state = RecorderState.boardFound;
          if (onStateChanged != null) onStateChanged!(_state);
          if (onBoardDetected != null) onBoardDetected!(boardState);
        }
      } else if (_state == RecorderState.boardFound ||
          _state == RecorderState.gameInProgress) {
        // Update piece types based on the current board
        _updatePieceTypes(boardState.pieces);

        // Detect moves
        final move = detectMoves(
          boardState.pieces,
          boardState,
          frame,
          frameTime,
        );
        if (move != null) {
          board.move(move);
          if (onMoveDetected != null) onMoveDetected!(move);

          if (_state == RecorderState.boardFound) {
            _state = RecorderState.gameInProgress;
            if (onStateChanged != null) onStateChanged!(_state);
          }
        }

        // Update previous frame and detection for capture move analysis
        prevFrame = img.clone(frame);
        prevDetection = {for (var p in boardState.pieces) p.square: p};
      }

      return boardState;
    } catch (e) {
      logger.e("Error processing frame: $e");
      return null;
    }
  }

  /// Update piece types based on the current board state
  void _updatePieceTypes(List<ChessPiece> pieces) {
    for (final piece in pieces) {
      try {
        final boardPiece = board.get(piece.square);
        if (boardPiece != null) {
          piece.pieceType = boardPiece.type.toUpperCase();
          if (!boardPiece.color) {
            // Black piece
            piece.pieceType = piece.pieceType?.toLowerCase();
          }
        } else {
          piece.pieceType = null;
        }
      } catch (e) {
        logger.e("Error updating piece type: $e");
      }
    }
  }

  /// Detect chess moves from detected pieces
  chess.Move? detectMoves(
    List<ChessPiece> detectedPieces,
    ChessboardState? boardState,
    img.Image? currentFrame,
    double frameTime,
  ) {
    if (detectedPieces.isEmpty) {
      return null;
    }

    // Skip move detection if we recently committed a move
    final moveBufferWindowSec = config.moveBufferWindowMs / 1000.0;
    if (lastCommittedMoveTime != null &&
        frameTime - lastCommittedMoveTime! < moveBufferWindowSec) {
      return null;
    }

    // Get current board squares and detected squares
    final detectedSquares = detectedPieces.map((p) => p.square).toSet();
    final currentSquares = <String>{};

    for (int i = 0; i < 64; i++) {
      final square = chess.Chess.algebraic(i);
      if (board.get(square) != null) {
        currentSquares.add(square);
      }
    }

    // Find differences
    final newSquares = detectedSquares.difference(currentSquares);
    final oldSquares = currentSquares.difference(detectedSquares);

    chess.Move? detectedMove;

    // Simple move analysis
    if (newSquares.length == 1 && oldSquares.length == 1) {
      final fromSquare = oldSquares.first;
      final toSquare = newSquares.first;

      final moveString = "$fromSquare$toSquare";
      chess.Move? move = _createMoveIfLegal(moveString);

      // Check for promotion
      if (move == null) {
        final piece = board.get(fromSquare);
        if (piece != null && piece.type.toLowerCase() == 'p') {
          final toRank = int.parse(toSquare[1]);
          if ((piece.color && toRank == 8) || (!piece.color && toRank == 1)) {
            // Find the piece at the destination square
            final promotedPiece = detectedPieces.firstWhereOrNull(
              (p) => p.square == toSquare,
            );

            if (promotedPiece != null && currentFrame != null) {
              // Extract the piece image from the frame
              final box = promotedPiece.box;
              final pieceImg = _extractPieceImage(
                currentFrame,
                box[0].toInt(),
                box[1].toInt(),
                box[2].toInt(),
                box[3].toInt(),
              );

              // Use classification model to predict piece type
              final pieceType = _classifyPiece(pieceImg);

              if (pieceType != null && pieceType != 'p' && pieceType != 'P') {
                // Create promotion move
                final promotionMove =
                    "$fromSquare$toSquare${pieceType.toLowerCase()}";
                move = _createMoveIfLegal(promotionMove);
              }
            }
          }
        }
      }

      if (move != null) {
        detectedMove = move;
      }
    }
    // Capture move analysis
    else if (newSquares.isEmpty &&
        oldSquares.length == 1 &&
        boardState != null &&
        currentFrame != null &&
        prevDetection.isNotEmpty) {
      final fromSquare = oldSquares.first;
      final movingPiece = board.get(fromSquare);

      if (movingPiece != null) {
        // Consider only capturing moves
        final candidateMoves = <chess.Move>[];

        // Get all legal moves from this square
        for (final move in board.generate_moves()) {
          if (move.fromAlgebraic == fromSquare &&
              board.isCapture(move.toAlgebraic)) {
            candidateMoves.add(move);
          }
        }

        chess.Move? bestCandidate;
        for (final move in candidateMoves) {
          final destSquare = move.toAlgebraic;
          if (prevDetection.containsKey(destSquare)) {
            final prevPiece = prevDetection[destSquare]!;
            // Find current detection for the destination square
            final currentPiece = detectedPieces.firstWhereOrNull(
              (p) => p.square == destSquare,
            );

            if (currentPiece != null &&
                prevFrame != null) {
              // Extract regions of interest for the pieces from previous and current frames
              final prevRoi = _extractPieceImage(
                prevFrame!,
                prevPiece.box[0].toInt(),
                prevPiece.box[1].toInt(),
                prevPiece.box[2].toInt(),
                prevPiece.box[3].toInt(),
              );

              final currentRoi = _extractPieceImage(
                currentFrame,
                currentPiece.box[0].toInt(),
                currentPiece.box[1].toInt(),
                currentPiece.box[2].toInt(),
                currentPiece.box[3].toInt(),
              );

              // Classify piece color
              final predCurrent = _classifyPieceColor(currentRoi);
              if (predCurrent == null) continue;

              // Determine the moving piece's color: false for white (uppercase), true for black
              final movingColor = !movingPiece.color;
              // For a valid capture, the piece at the destination should be of the opposite color
              if (predCurrent == movingColor) continue;

              // Check for promotion captures
              final toRank = int.parse(destSquare[1]);
              if ((movingPiece.type.toLowerCase() == 'p') &&
                  ((movingPiece.color && toRank == 8) ||
                      (!movingPiece.color && toRank == 1))) {
                // Classify the piece type
                final pieceType = _classifyPiece(currentRoi);
                if (pieceType != null && pieceType != 'p' && pieceType != 'P') {
                  // Create promotion capture move
                  final promotionMove =
                      "$fromSquare$destSquare${pieceType.toLowerCase()}";
                  bestCandidate = _createMoveIfLegal(promotionMove);
                  if (bestCandidate != null) break;
                }
              } else {
                bestCandidate = move;
              }
            }
          }
        }

        if (bestCandidate != null) {
          detectedMove = bestCandidate;
        }
      }
    }
    // En passant move analysis
    else if (oldSquares.length == 2 && newSquares.length == 1) {
      final toSquare = newSquares.first;
      final oldSquaresList = oldSquares.toList();

      // Find which old square has our pawn (the capturing piece)
      String? capturingSquare;
      String? capturedSquare;

      for (final square in oldSquaresList) {
        final piece = board.get(square);
        if (piece != null && piece.type.toLowerCase() == 'p') {
          if (piece.color == board.turn) {
            capturingSquare = square;
          } else {
            capturedSquare = square;
          }
        }
      }

      if (capturingSquare != null && capturedSquare != null) {
        final moveString = "$capturingSquare$toSquare";
        final move = _createMoveIfLegal(moveString);
        if (move != null) {
          detectedMove = move;
        }
      }
    }
    // Castling move analysis
    else if (newSquares.length == 2 && oldSquares.length == 2) {
      final newSquaresList = newSquares.toList()..sort();
      final oldSquaresList = oldSquares.toList()..sort();

      // Define castling patterns for both colors
      final castlingPatterns = {
        // White castling patterns
        'white_kingside': {
          'old': ['e1', 'h1'], // king and rook original squares
          'new': ['f1', 'g1'], // king and rook final squares
        },
        'white_queenside': {
          'old': ['a1', 'e1'],
          'new': ['c1', 'd1'],
        },
        // Black castling patterns
        'black_kingside': {
          'old': ['e8', 'h8'],
          'new': ['f8', 'g8'],
        },
        'black_queenside': {
          'old': ['a8', 'e8'],
          'new': ['c8', 'd8'],
        },
      };

      // Check if the squares match any castling pattern based on the current turn
      chess.Move? castlingMove;
      if (board.turn) {
        // White's turn
        if (_listsEqual(
              oldSquaresList,
              castlingPatterns['white_kingside']!['old']!,
            ) &&
            _listsEqual(
              newSquaresList,
              castlingPatterns['white_kingside']!['new']!,
            )) {
          castlingMove = _createMoveIfLegal('e1g1');
        } else if (_listsEqual(
              oldSquaresList,
              castlingPatterns['white_queenside']!['old']!,
            ) &&
            _listsEqual(
              newSquaresList,
              castlingPatterns['white_queenside']!['new']!,
            )) {
          castlingMove = _createMoveIfLegal('e1c1');
        }
      } else {
        // Black's turn
        if (_listsEqual(
              oldSquaresList,
              castlingPatterns['black_kingside']!['old']!,
            ) &&
            _listsEqual(
              newSquaresList,
              castlingPatterns['black_kingside']!['new']!,
            )) {
          castlingMove = _createMoveIfLegal('e8g8');
        } else if (_listsEqual(
              oldSquaresList,
              castlingPatterns['black_queenside']!['old']!,
            ) &&
            _listsEqual(
              newSquaresList,
              castlingPatterns['black_queenside']!['new']!,
            )) {
          castlingMove = _createMoveIfLegal('e8c8');
        }
      }

      if (castlingMove != null) {
        detectedMove = castlingMove;
      }
    }

    // If we detected a move, add it to the buffer
    if (detectedMove != null) {
      moveBuffer.add(Tuple2(frameTime, detectedMove));

      // Try to process the buffer
      final committedMove = _processMoveBuffer(frameTime);
      if (committedMove != null) {
        lastCommittedMoveTime = frameTime;
        lastCommittedMove = committedMove;
        return committedMove;
      }
    }

    return null;
  }

  /// Process the move buffer to find the most consistent move detection
  chess.Move? _processMoveBuffer(double frameTime) {
    // Remove old moves outside the time window
    final moveBufferWindowSec = config.moveBufferWindowMs / 1000.0;
    final cutoffTime = frameTime - moveBufferWindowSec;
    moveBuffer.removeWhere((item) => item.item1 <= cutoffTime);

    if (moveBuffer.isEmpty) {
      return null;
    }

    // Count move occurrences
    final moveCounts = <String, int>{};
    for (final item in moveBuffer) {
      final moveString = item.item2.san;
      moveCounts[moveString] = (moveCounts[moveString] ?? 0) + 1;
    }

    // Find the most common move
    String? mostCommonMove;
    int maxCount = 0;

    moveCounts.forEach((move, count) {
      if (count > maxCount) {
        maxCount = count;
        mostCommonMove = move;
      }
    });

    // Check if we have enough consistent detections
    if (mostCommonMove != null && maxCount >= config.minMoveDetections) {
      // Clear the buffer after committing a move
      moveBuffer.clear();

      // Find the actual move object
      final moveObj =
          moveBuffer
              .firstWhereOrNull((item) => item.item2.san == mostCommonMove)
              ?.item2;

      return moveObj;
    }

    return null;
  }

  /// Create a Move object if the move is legal
  chess.Move? _createMoveIfLegal(String moveString) {
    try {
      final move = chess.Move.fromUci(moveString);
      if (board.isLegalMove(move)) {
        return move;
      }
    } catch (e) {
      logger.e("Invalid move string: $moveString");
    }
    return null;
  }

  /// Compare two lists for equality (ignoring order)
  bool _listsEqual<T>(List<T> a, List<T> b) {
    if (a.length != b.length) return false;

    final sortedA = List<T>.from(a)..sort();
    final sortedB = List<T>.from(b)..sort();

    for (int i = 0; i < sortedA.length; i++) {
      if (sortedA[i] != sortedB[i]) return false;
    }

    return true;
  }

  /// Extract piece image from frame
  img.Image _extractPieceImage(
    img.Image frame,
    int x1,
    int y1,
    int x2,
    int y2,
  ) {
    // Ensure coordinates are within image bounds
    x1 = math.max(0, math.min(x1, frame.width - 1));
    y1 = math.max(0, math.min(y1, frame.height - 1));
    x2 = math.max(0, math.min(x2, frame.width - 1));
    y2 = math.max(0, math.min(y2, frame.height - 1));

    // Ensure width and height are positive
    if (x2 <= x1) x2 = x1 + 1;
    if (y2 <= y1) y2 = y1 + 1;

    return img.copyCrop(frame, x: x1, y: y1, width: x2 - x1, height: y2 - y1);
  }

  /// Classify piece type using the piece classifier model
  String? _classifyPiece(img.Image pieceImg) {
    if (pieceClassifierModel == null) return null;

    try {
      // Resize to model input dimensions
      final modelInputSize = 224; // Adjust to match your model's input size
      final resizedImg = img.copyResize(
        pieceImg,
        width: modelInputSize,
        height: modelInputSize,
        interpolation: img.Interpolation.cubic,
      );

      // Convert to input tensor format
      final inputBuffer = Float32List(1 * modelInputSize * modelInputSize * 3);
      int pixelIndex = 0;

      for (int y = 0; y < modelInputSize; y++) {
        for (int x = 0; x < modelInputSize; x++) {
          final pixel = resizedImg.getPixel(x, y);
          // Normalize [0, 255] to [0, 1]
          inputBuffer[pixelIndex] = img.getRed(pixel) / 255.0;
          inputBuffer[pixelIndex + 1] = img.getGreen(pixel) / 255.0;
          inputBuffer[pixelIndex + 2] = img.getBlue(pixel) / 255.0;
          pixelIndex += 3;
        }
      }

      // Reshape for the model
      final input = inputBuffer.reshape([1, modelInputSize, modelInputSize, 3]);

      // Prepare output tensor
      final outputBuffer = List.filled(1 * 5, 0.0).reshape([1, 5]);

      // Run inference
      pieceClassifierModel!.run(input, outputBuffer);

      // Process results
      int maxIndex = 0;
      double maxValue = outputBuffer[0][0];

      for (int i = 1; i < 5; i++) {
        if (outputBuffer[0][i] > maxValue) {
          maxValue = outputBuffer[0][i];
          maxIndex = i;
        }
      }

      if (maxValue < 0.5) return null; // Low confidence

      // Map class index to piece type
      // Adjust these indices based on your model's classes
      final pieceTypes = ['b', 'k', 'p', 'q', 'r'];
      if (maxIndex >= 0 && maxIndex < pieceTypes.length) {
        // Return uppercase for white pieces, lowercase for black
        // This is determined by the color classifier
        return pieceTypes[maxIndex];
      }

      return null;
    } catch (e) {
      logger.e("Error classifying piece: $e");
      return null;
    }
  }

  /// Classify piece color using the color classifier model
  bool? _classifyPieceColor(img.Image pieceImg) {
    if (colorClassifierModel == null) return null;

    try {
      // Resize to model input dimensions
      final modelInputSize = 75; // Adjust to match your model's input size
      final resizedImg = img.copyResize(
        pieceImg,
        width: modelInputSize,
        height: modelInputSize,
        interpolation: img.Interpolation.cubic,
      );

      // Convert to input tensor format
      final inputBuffer = Float32List(1 * modelInputSize * modelInputSize * 3);
      int pixelIndex = 0;

      for (int y = 0; y < modelInputSize; y++) {
        for (int x = 0; x < modelInputSize; x++) {
          final pixel = resizedImg.getPixel(x, y);
          // Normalize using same values as your PyTorch model
          // [0, 255] to [-1, 1] using mean=0.5, std=0.5
          inputBuffer[pixelIndex] = (img.getRed(pixel) / 255.0 - 0.5) / 0.5;
          inputBuffer[pixelIndex + 1] =
              (img.getGreen(pixel) / 255.0 - 0.5) / 0.5;
          inputBuffer[pixelIndex + 2] =
              (img.getBlue(pixel) / 255.0 - 0.5) / 0.5;
          pixelIndex += 3;
        }
      }

      // Reshape for the model
      final input = inputBuffer.reshape([1, modelInputSize, modelInputSize, 3]);

      // Prepare output tensor
      final outputBuffer = List.filled(1 * 2, 0.0).reshape([1, 2]);

      // Run inference
      colorClassifierModel!.run(input, outputBuffer);

      // Process results: 0 for white, 1 for black
      return outputBuffer[0][1] > outputBuffer[0][0];
    } catch (e) {
      logger.e("Error classifying piece color: $e");
      return null;
    }
  }

  /// Get the current FEN notation
  String getCurrentFen() {
    return board.fen;
  }

  /// Reset the game to the initial position
  void resetGame() {
    board = chess.Chess();

    if (config.initialPosition != null) {
      try {
        board.load(config.initialPosition!);
      } catch (e) {
        logger.e(
          "Invalid FEN provided for start position. Using standard starting position.",
        );
      }
    }

    trackedPieces.clear();
    prevFrame = null;
    prevDetection.clear();
    moveBuffer.clear();
    lastCommittedMoveTime = null;
    lastCommittedMove = null;

    _state = RecorderState.findingBoard;
    if (onStateChanged != null) onStateChanged!(_state);

    logger.i("Game reset to initial position");
  }

  /// Clean up resources
  void dispose() {
    pieceDetectorModel?.close();
    pieceClassifierModel?.close();
    colorClassifierModel?.close();
    logger.i("Chess recorder disposed");
  }
}
