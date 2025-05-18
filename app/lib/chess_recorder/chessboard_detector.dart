import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:chess/chess.dart' as chess;
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tuple/tuple.dart';
import 'package:logger/logger.dart';
import 'package:collection/collection.dart';
import 'dart:async';
import 'dart:io';

import 'corner_detector.dart';
import 'chess_models.dart';

class BoardDetectionException implements Exception {
  final String message;
  BoardDetectionException(this.message);
  
  @override
  String toString() => 'BoardDetectionException: $message';
}

class ChessboardDetector {
  final Logger logger = Logger();
  final Interpreter? pieceDetectorModel;
  final CornerDetector cornerDetector;
  final ConfigNode config;
  final bool debugMode;
  
  // Cached corner data
  List<List<double>>? cachedCorners;
  List<List<double>>? cachedHomography;
  List<int>? cachedDims;
  double cornerQualityScore = 0.90;
  bool forceCornerDetection = false;
  
  // Current board state
  chess.Chess? currentBoard;
  
  ChessboardDetector({
    required this.pieceDetectorModel,
    ConfigNode? config,
    this.debugMode = false,
  }) : 
    cornerDetector = CornerDetector(config: config),
    config = config ?? ConfigNode();
  
  /// Detect pieces using TFLite model
  Future<Tuple2<List<List<double>>, List<double>>> detectPieces(cv.Mat image) async {
    if (pieceDetectorModel == null) {
      throw BoardDetectionException("Piece detector model not initialized");
    }
    
    try {
      // Resize to model input dimensions
      // These dimensions should match what your model expects
      final modelInputWidth = 640; // Adjust to your model's input size
      final modelInputHeight = 640; // Adjust to your model's input size

      final inputMat = cv.resize(image, (modelInputHeight, modelInputWidth));
      
      
      // Prepare output tensors
      // These sizes should match what your model produces
      // For YOLO model: [1, 8400, 6] where 6 is [x, y, width, height, confidence, class]
      final outputBoxes = List.filled(1 * 8400 * 6, 0.0).reshape([1, 8400, 6]);
      
      // Run inference
      pieceDetectorModel!.run(inputMat, outputBoxes);
      
      // Process results - extract bounding boxes and confidences
      final boxes = <List<double>>[];
      final confidences = <double>[];
      
      // Process model output to extract boxes
      // This processing depends on your model's output format
      for (int i = 0; i < 8400; i++) {
        final confidence = outputBoxes[0][i][4];
        if (confidence > 0.5) { // Confidence threshold
          final x = outputBoxes[0][i][0] * image.width / modelInputWidth;
          final y = outputBoxes[0][i][1] * image.height / modelInputHeight;
          final w = outputBoxes[0][i][2] * image.width / modelInputWidth;
          final h = outputBoxes[0][i][3] * image.height / modelInputHeight;
          
          // Convert to [x1, y1, x2, y2] format
          boxes.add([x - w/2, y - h/2, x + w/2, y + h/2]);
          confidences.add(confidence);
        }
      }
      
      return Tuple2(boxes, confidences);
    } catch (e) {
      logger.e("Error detecting pieces: $e");
      throw BoardDetectionException("Failed to detect pieces: $e");
    }
  }
  
  /// Validate corner detection quality
  double validateCorners(List<List<double>> boxes, List<List<double>> corners, 
      List<List<double>> homography, List<int> dims) {
    try {
      // Warp piece boxes to check if they fall within the board
      final warpedBoxes = <List<double>>[];
      
      for (final box in boxes) {
        final points = [
          [box[0], box[1]],  // top-left
          [box[2], box[1]],  // top-right
          [box[2], box[3]],  // bottom-right
          [box[0], box[3]]   // bottom-left
        ];
        
        final warpedPoints = _applyHomography(points, homography);
        
        // Get bounding box of warped points
        double x1 = double.infinity, y1 = double.infinity;
        double x2 = -double.infinity, y2 = -double.infinity;
        
        for (final point in warpedPoints) {
          x1 = math.min(x1, point[0]);
          y1 = math.min(y1, point[1]);
          x2 = math.max(x2, point[0]);
          y2 = math.max(y2, point[1]);
        }
        
        warpedBoxes.add([x1, y1, x2, y2]);
      }
      
      // Warp corners
      final warpedCorners = _applyHomography(corners, homography);
      
      // Create chess grid
      final squares = _createChessGrid(warpedCorners, blackOnLeft: true);
      
      // Count pieces that map to valid squares
      int validMappings = 0;
      int boardStateMatches = 0;
      final totalPieces = warpedBoxes.length;
      
      // Get current board piece positions if available
      final boardPieces = <String, bool>{};
      if (currentBoard != null) {
        for (int i = 0; i < 64; i++) {
          final square = chess.Chess.algebraic(i);
          final piece = currentBoard!.get(square);
          if (piece != null) {
            boardPieces[square] = true;
          }
        }
      }

      final mappedSquares = <String>{};
      
      for (final warpedBox in warpedBoxes) {
        // Use bottom half of warped box for mapping
        final midY = (warpedBox[1] + warpedBox[3]) / 2;
        final boxBottom = [
          [warpedBox[0], midY],
          [warpedBox[2], midY],
          [warpedBox[2], warpedBox[3]],
          [warpedBox[0], warpedBox[3]]
        ];
        
        String? bestSquare;
        double maxOverlap = 0;
        
        // Find square with highest overlap
        for (final square in squares) {
          final coords = square.item1;
          final name = square.item2;
          
          final overlap = _calculateOverlap(boxBottom, coords);
          if (overlap > 0.5 && overlap > maxOverlap) { // 50% overlap threshold
            maxOverlap = overlap;
            bestSquare = name;
          }
        }
        
        if (bestSquare != null) {
          validMappings++;
          mappedSquares.add(bestSquare);
          
          // Check if this mapping matches current board state
          if (currentBoard != null && boardPieces.containsKey(bestSquare)) {
            boardStateMatches++;
          }
        }
      }
      
      // Calculate quality scores
      final mappingScore = totalPieces > 0 ? validMappings / totalPieces : 0.0;
      
      // Calculate board state matching score
      double boardScore = 1.0;
      if (currentBoard != null) {
        final expectedPieces = boardPieces.length;
        if (expectedPieces > 0) {
          // Consider both accuracy and completeness of detection
          final precision = totalPieces > 0 ? boardStateMatches / totalPieces : 0.0;
          final recall = boardStateMatches / expectedPieces;
          boardScore = (precision + recall) / 2;
        }
      }
      
      // Combine scores (give more weight to board state matching)
      final finalScore = (mappingScore * 0.4) + (boardScore * 0.6);
      
      if (debugMode) {
        logger.d("Corner validation - Mapping score: ${mappingScore.toStringAsFixed(2)}, "
          "Board score: ${boardScore.toStringAsFixed(2)}, Final score: ${finalScore.toStringAsFixed(2)}");
      }
      
      return finalScore;
    } catch (e) {
      logger.e("Corner validation failed: $e");
      return 0.0;
    }
  }
  
  /// Apply homography transformation to points
  List<List<double>> _applyHomography(List<List<double>> points, List<List<double>> homography) {
    final warpedPoints = <List<double>>[];
    
    for (final point in points) {
      // Add homogeneous coordinate
      final homogeneousPoint = [point[0], point[1], 1.0];
      
      // Apply transformation
      double x = homography[0][0] * homogeneousPoint[0] +
                 homography[0][1] * homogeneousPoint[1] +
                 homography[0][2] * homogeneousPoint[2];
      
      double y = homography[1][0] * homogeneousPoint[0] +
                 homography[1][1] * homogeneousPoint[1] +
                 homography[1][2] * homogeneousPoint[2];
      
      double w = homography[2][0] * homogeneousPoint[0] +
                 homography[2][1] * homogeneousPoint[1] +
                 homography[2][2] * homogeneousPoint[2];
      
      // Normalize by w
      warpedPoints.add([x / w, y / w]);
    }
    
    return warpedPoints;
  }
  
  /// Create 8x8 grid coordinates for the chessboard
  List<Tuple2<List<List<double>>, String>> _createChessGrid(
      List<List<double>> warpedCorners, {bool blackOnLeft = false}) {
    
    // Create evenly spaced coordinates along x and y axes
    final xCoords = List.generate(9, (i) => warpedCorners[0][0] + 
        (warpedCorners[1][0] - warpedCorners[0][0]) * i / 8);
    
    final yCoords = List.generate(9, (i) => warpedCorners[0][1] + 
        (warpedCorners[3][1] - warpedCorners[0][1]) * i / 8);
    
    final squares = <Tuple2<List<List<double>>, String>>[];
    
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        final square = [
          [xCoords[j], yCoords[i]],
          [xCoords[j+1], yCoords[i]],
          [xCoords[j+1], yCoords[i+1]],
          [xCoords[j], yCoords[i+1]]
        ];
        
        String squareName;
        if (blackOnLeft) {
          // Black's perspective (a1 at bottom right)
          squareName = "${String.fromCharCode(97 + (7 - j))}${8 - i}";
        } else {
          // White's perspective (a1 at top left)
          squareName = "${String.fromCharCode(97 + j)}${i + 1}";
        }
        
        squares.add(Tuple2(square, squareName));
      }
    }
    
    return squares;
  }
  
  /// Calculate overlap between two polygons (simplified)
  double _calculateOverlap(List<List<double>> poly1, List<List<double>> poly2) {
    // This is a simplified overlap calculation
    // In a real implementation, you would use a proper polygon intersection algorithm
    
    // For now, we'll approximate by checking the center of poly1 is inside poly2
    final centerX = poly1.map((p) => p[0]).reduce((a, b) => a + b) / poly1.length;
    final centerY = poly1.map((p) => p[1]).reduce((a, b) => a + b) / poly1.length;
    
    // Check if the center point is inside poly2
    if (_pointInPolygon([centerX, centerY], poly2)) {
      return 0.7; // Approximate overlap
    }
    
    return 0.0;
  }
  
  /// Check if a point is inside a polygon
  bool _pointInPolygon(List<double> point, List<List<double>> polygon) {
    // Ray casting algorithm to determine if point is in polygon
    bool inside = false;
    int i, j;
    for (i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      if (((polygon[i][1] > point[1]) != (polygon[j][1] > point[1])) &&
          (point[0] < (polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) /
           (polygon[j][1] - polygon[i][1]) + polygon[i][0])) {
        inside = !inside;
      }
    }
    return inside;
  }
  
  /// Process a frame and return the chess state
  Future<ChessboardState> processFrame(cv.Mat frame, {chess.Chess? currentBoard}) async {
    // Update current board reference
    this.currentBoard = currentBoard;
    
    try {
      // Resize image for processing
      final resizedResult = cornerDetector.resizeImage(frame);
      final resizedFrame = resizedResult.item1;
      final scale = resizedResult.item2;
      
      // Detect pieces
      final detectResult = await detectPieces(resizedFrame);
      final boxes = detectResult.item1;
      final confidences = detectResult.item2;
      
      List<List<double>> corners;
      List<List<double>> homography;
      List<int> dims;
      
      // Detect board and get transformation
      if (forceCornerDetection || cachedCorners == null) {
        final cornerResult = await cornerDetector.findCorners(resizedFrame);
        corners = cornerResult.item1;
        homography = cornerResult.item2;
        dims = cornerResult.item3;
        
        // Validate new corner detection
        final qualityScore = validateCorners(boxes, corners, homography, dims);
        if (forceCornerDetection || cachedCorners == null || qualityScore > cornerQualityScore) {
          cachedCorners = corners;
          cachedHomography = homography;
          cachedDims = dims;
          cornerQualityScore = qualityScore;
          logger.i("Updated corner cache with quality score: ${qualityScore.toStringAsFixed(2)}");
          forceCornerDetection = false;
        }
      } else {
        // Use cached corners
        corners = cachedCorners!;
        homography = cachedHomography!;
        dims = cachedDims!;
      }
      
      // Create debug images
      final cornerDebugImage = _createCornerDebugImage(resizedFrame, corners);
      final pieceDetectionImage = _createPieceDetectionImage(resizedFrame, boxes, confidences);
      
      // Warp image
      final warpedImage = _warpImage(resizedFrame, homography, dims);
      
      // Warp corners
      final warpedCorners = _applyHomography(corners, homography);
      
      // Warp piece boxes for square mapping
      final warpedBoxes = boxes.map((box) {
        // Convert box to points
        final points = [
          [box[0], box[1]],  // top-left
          [box[2], box[1]],  // top-right
          [box[2], box[3]],  // bottom-right
          [box[0], box[3]]   // bottom-left
        ];
        
        final warpedPoints = _applyHomography(points, homography);
        
        // Get bounding box of warped points
        double x1 = double.infinity, y1 = double.infinity;
        double x2 = -double.infinity, y2 = -double.infinity;
        
        for (final point in warpedPoints) {
          x1 = math.min(x1, point[0]);
          y1 = math.min(y1, point[1]);
          x2 = math.max(x2, point[0]);
          y2 = math.max(y2, point[1]);
        }
        
        return [x1, y1, x2, y2];
      }).toList();
      
      // Create chess grid
      final squares = _createChessGrid(warpedCorners, blackOnLeft: true);
      
      // Map pieces to squares
      final pieces = <ChessPiece>[];
      final debugImage = img.clone(resizedFrame);
      
      // First pass: Calculate overlaps for all pieces
      final squareMappings = <String, List<Tuple3<List<double>, double, double>>>{};
      
      for (int i = 0; i < boxes.length; i++) {
        final box = boxes[i];
        final conf = confidences[i];
        final warpedBox = warpedBoxes[i];
        
        // Use bottom half of warped box
        final midY = (warpedBox[1] + warpedBox[3]) / 2;
        final boxBottom = [
          [warpedBox[0], midY],
          [warpedBox[2], midY],
          [warpedBox[2], warpedBox[3]],
          [warpedBox[0], warpedBox[3]]
        ];
        
        // Find square with highest overlap ratio
        String? bestSquare;
        double maxOverlap = 0;
        
        for (final square in squares) {
          final coords = square.item1;
          final name = square.item2;
          
          final overlap = _calculateOverlap(boxBottom, coords);
          if (overlap > maxOverlap) {
            maxOverlap = overlap;
            bestSquare = name;
          }
        }
        
        if (bestSquare != null) {
          if (!squareMappings.containsKey(bestSquare)) {
            squareMappings[bestSquare] = [];
          }
          squareMappings[bestSquare]!.add(Tuple3(box, conf, maxOverlap));
        }
      }
      
      // Second pass: Select best piece for each square
      for (final entry in squareMappings.entries) {
        final squareName = entry.key;
        final candidates = entry.value;
        
        if (candidates.isNotEmpty) {
          // Sort by overlap ratio and take the highest
          candidates.sort((a, b) => b.item3.compareTo(a.item3));
          final bestCandidate = candidates.first;
          final box = bestCandidate.item1;
          final conf = bestCandidate.item2;
          
          // Add piece with original box coordinates
          pieces.add(ChessPiece(
            square: squareName,
            box: Float64List.fromList(box),
            confidence: conf,
          ));
          
          // Draw debug visualization
          _drawBoxOnImage(debugImage, box, squareName, conf);
        }
      }
      
      return ChessboardState(
        pieces: pieces,
        warpedImage: warpedImage,
        debugImage: debugImage,
        cornerDebugImage: cornerDebugImage,
        pieceDetectionImage: pieceDetectionImage,
      );
    } catch (e) {
      logger.e("Failed to process frame: $e");
      throw BoardDetectionException("Failed to process frame: $e");
    }
  }
  
  /// Create corner debug image
  img.Image _createCornerDebugImage(img.Image frame, List<List<double>> corners) {
    final debugImage = img.clone(frame);
    
    for (int i = 0; i < corners.length; i++) {
      final x = corners[i][0].toInt();
      final y = corners[i][1].toInt();
      
      // Draw circle at corner
      img.drawCircle(
        debugImage,
        x: x,
        y: y,
        radius: 5,
        color: img.ColorRgb8(0, 255, 0),
        antialias: true,
      );
      
      // Add label
      img.drawString(
        debugImage,
        string: "${i+1}",
        x: x + 10,
        y: y + 10,
        color: img.ColorRgb8(0, 255, 0),
        font: img.arial24,
      );
    }
    
    // Add quality score text
    if (debugMode) {
      final qualityText = "Corner Quality: ${cornerQualityScore.toStringAsFixed(2)}";
      img.drawString(
        debugImage,
        string: qualityText,
        x: 10,
        y: 30,
        color: img.ColorRgb8(0, 255, 0),
        font: img.arial24,
      );
    }
    
    return debugImage;
  }
  
  /// Create piece detection debug image
  img.Image _createPieceDetectionImage(img.Image frame, List<List<double>> boxes, List<double> confidences) {
    final debugImage = img.clone(frame);
    
    for (int i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      final conf = confidences[i];
      
      // Draw rectangle
      img.drawRect(
        debugImage,
        x1: box[0].toInt(),
        y1: box[1].toInt(),
        x2: box[2].toInt(),
        y2: box[3].toInt(),
        color: img.ColorRgb8(0, 0, 255),
        thickness: 2,
      );
      
      // Add confidence score
      img.drawString(
        debugImage,
        string: conf.toStringAsFixed(2),
        x: box[0].toInt(),
        y: box[1].toInt() - 5,
        color: img.ColorRgb8(0, 0, 255),
        font: img.arial14,
      );
    }
    
    return debugImage;
  }
  
  /// Warp image using homography
  img.Image _warpImage(img.Image frame, List<List<double>> homography, List<int> dims) {
    // This is a simplified implementation
    // A real implementation would use proper perspective transform
    // which would be more efficiently done with platform channels to native code
    
    final width = dims[0];
    final height = dims[1];
    final warpedImage = img.Image(width: width, height: height);
    
    // In a real implementation, you would apply the homography transformation
    // For now, we're just returning a resized version of the original image
    return img.copyResize(frame, width: width, height: height);
  }
  
  /// Draw box with label on image
  void _drawBoxOnImage(img.Image image, List<double> box, String label, double confidence) {
    // Draw rectangle
    img.drawRect(
      image,
      x1: box[0].toInt(),
      y1: box[1].toInt(),
      x2: box[2].toInt(),
      y2: box[3].toInt(),
      color: img.ColorRgb8(0, 255, 0),
      thickness: 2,
    );
    
    // Add label with confidence
    img.drawString(
      image,
      string: "$label (${confidence.toStringAsFixed(2)})",
      x: box[0].toInt(),
      y: box[1].toInt() - 5,
      color: img.ColorRgb8(0, 255, 0),
      font: img.arial14,
    );
    
    // Draw overlap ratio for debugging
    if (debugMode) {
      img.drawString(
        image,
        string: "o:0.70", // Placeholder for the overlap calculation
        x: box[0].toInt(),
        y: box[3].toInt() + 15,
        color: img.ColorRgb8(255, 0, 0),
        font: img.arial14,
      );
    }
  }
  
  /// Force a new corner detection on the next frame
  void forceCornerDetectionUpdate() {
    forceCornerDetection = true;
  }
} 