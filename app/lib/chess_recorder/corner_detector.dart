import 'dart:typed_data';
import 'dart:math' as math;
import 'package:collection/collection.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:tuple/tuple.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/foundation.dart';
import 'package:logger/logger.dart';
import 'package:simple_cluster/simple_cluster.dart';

/// Placeholder for a 3x3 matrix. Replace with your actual matrix type if needed.
typedef HomographyMatrix = List<List<double>>;

class ConfigNode {
  final double bestSolutionTolerance;
  final double offsetTolerance;
  final int boardSizeMin;
  final int boardSizeMax;
  final int harrisBlockSize;
  final double harrisK;
  final int harrisQuality;
  final int cannyThresh1;
  final int cannyThresh2;
  final int houghThreshold;
  final int houghMinLineLength;
  final int houghMaxLineGap;

  var cannyAperture;

  ConfigNode({
    this.bestSolutionTolerance = 0.1,
    this.offsetTolerance = 0.1,
    this.boardSizeMin = 100,
    this.boardSizeMax = 1000,
    this.harrisBlockSize = 2,
    this.harrisK = 0.04,
    this.harrisQuality = 30,
    this.cannyThresh1 = 50,
    this.cannyThresh2 = 200,
    this.houghThreshold = 50,
    this.houghMinLineLength = 50,
    this.houghMaxLineGap = 10,
    this.cannyAperture = 3,
  });
}

class CornerDetectionException implements Exception {
  final String message;
  CornerDetectionException(this.message);

  @override
  String toString() => 'CornerDetectionException: $message';
}

class CornerDetector {
  final ConfigNode config;
  final Logger logger = Logger();

  CornerDetector({ConfigNode? config}) : config = config ?? ConfigNode();

  /// Resize image to appropriate dimensions
  Tuple2<cv.Mat, double> resizeImage(cv.Mat image) {
    final width = image.width;
    final height = image.height;

    if (width == config.boardSizeMax) {
      return Tuple2(image, 1.0);
    }

    final scale = config.boardSizeMax / width;
    final newWidth = (width * scale).round();
    final newHeight = (height * scale).round();

    final resizedImage = cv.resize(image, (newHeight, newWidth));

    return Tuple2(resizedImage, scale);
  }

  // /// Detect Harris corners in the image
  // List<Tuple2<int, int>> detectHarrisCorners(img.Image image) {}

  // def _detect_edges(edge_detection_cfg: CN, gray: np.ndarray) -> np.ndarray:
  //   if gray.dtype != np.uint8:
  //       gray = gray / gray.max() * 255
  //       gray = gray.astype(np.uint8)
  //   edges = cv2.Canny(gray,
  //                     edge_detection_cfg.LOW_THRESHOLD,
  //                     edge_detection_cfg.HIGH_THRESHOLD,
  //                     edge_detection_cfg.APERTURE)
  //   return edges

  cv.Mat detectEdges(cv.Mat image) {
    final edges = cv.canny(
      image,
      config.cannyThresh1.toDouble(),
      config.cannyThresh2.toDouble(),
      apertureSize: config.cannyAperture,
    );
    return edges;
  }

  // def _detect_lines(cfg: CN, edges: np.ndarray) -> np.ndarray:
  //   # array of [rho, theta]
  //   lines = cv2.HoughLines(edges, 1, np.pi/360, cfg.LINE_DETECTION.THRESHOLD)
  //   lines = lines.squeeze(axis=-2)
  //   lines = _fix_negative_rho_in_hesse_normal_form(lines)

  //   if cfg.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION:
  //       threshold = np.deg2rad(
  //           cfg.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES)
  //       vmask = np.abs(lines[:, 1]) < threshold
  //       hmask = np.abs(lines[:, 1] - np.pi / 2) < threshold
  //       mask = vmask | hmask
  //       lines = lines[mask]
  //   return lines

  // def _fix_negative_rho_in_hesse_normal_form(lines: np.ndarray) -> np.ndarray:
  //   lines = lines.copy()
  //   neg_rho_mask = lines[..., 0] < 0
  //   lines[neg_rho_mask, 0] = - \
  //       lines[neg_rho_mask, 0]
  //   lines[neg_rho_mask, 1] =  \
  //       lines[neg_rho_mask, 1] - np.pi
  //   return lines

  /// Find the chess board corners
  Future<Tuple3<List<List<double>>, List<List<double>>, List<int>>> findCorners(
    cv.Mat image,
  ) async {
    try {
      // In a complete implementation, we would:
      // 1. Convert to grayscale
      // 2. Apply Canny edge detection
      // 3. Apply Hough line transform
      // 4. Find line intersections for corners
      // 5. Use RANSAC to find the best chessboard corners

      final gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY);
      // For now, we're using a simplified approach
      final edges = detectEdges(gray);

      final lines = detectLines(edges);

      if (lines.length > 400) {
        throw ChessboardNotLocatedException("Too many lines in the image");
      }
      final clusteredLines = clusterHorizontalAndVerticalLines(lines);
      var horizontalLines = clusteredLines.item1;
      var verticalLines = clusteredLines.item2;

      horizontalLines = eliminateSimilarLines(
        horizontalLines,
        verticalLines,
      );
      verticalLines = eliminateSimilarLines(verticalLines, horizontalLines);

      final intersectionPoints = getIntersectionPoints(horizontalLines, verticalLines);

      int bestNumInliers = 0;
      var bestConfiguration; // We'll define a proper structure later
      int iterations = 0;

      while (iterations < 200 || bestNumInliers < 30) {
        final rowIndices = chooseFromRange(horizontalLines.length);
        final colIndices = chooseFromRange(verticalLines.length);

        int row1 = rowIndices[0];
        int row2 = rowIndices[1];
        int col1 = colIndices[0];
        int col2 = colIndices[1];

        final homography = computeHomography(intersectionPoints, row1, row2, col1, col2);
        final warpedPoints = warpPoints(homography, intersectionPoints);
      }
 
      // Convert to the format expected by the rest of the code
      final cornersList =
          corners
              .map(
                (corner) => [corner.item1.toDouble(), corner.item2.toDouble()],
              )
              .toList();

      // Create a placeholder homography matrix (3x3 identity matrix)
      final homography = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
      ];

      // Dimensions for the transformed image
      final dims = [image.width, image.height];

      return Tuple3(cornersList, homography, dims);
    } catch (e) {
      logger.e('Error in corner detection: $e');
      throw CornerDetectionException('Failed to detect corners: $e');
    }
  }

  /// Computes a homography from 4 corner points to a regular rectangle.
  cv.Mat computeHomography(
    List<List<List<double>>> allIntersectionPoints,
    int row1,
    int row2,
    int col1,
    int col2,
  ) {
  final topLeft = allIntersectionPoints[row1][col1];
  final topRight = allIntersectionPoints[row1][col2];
  final bottomLeft = allIntersectionPoints[row2][col1];
  final bottomRight = allIntersectionPoints[row2][col2];

  final src = [
    topLeft,
    topRight,
    bottomRight,
    bottomLeft,
  ];

  final dst = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
  ];


  final srcMat = cv.Mat.from2DList(src, cv.MatType.CV_64FC4);
  final dstMat = cv.Mat.from2DList(dst, cv.MatType.CV_64FC4);
  return cv.findHomography(srcMat, dstMat);
}

List<List<List<double>>> warpPoints(cv.Mat homography, List<List<List<double>>> points) {
  final rows = points.length;
  final cols = points[0].length;

  // Convert homography to a 2D List and transpose it
  final H = homography.toList(); // shape [3][3]
  final Ht = List.generate(3, (i) => List.generate(3, (j) => H[j][i]));

  List<List<List<double>>> warpedPoints = List.generate(rows, (_) => List.filled(cols, [0.0, 0.0]));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      final x = points[i][j][0];
      final y = points[i][j][1];

      final x_ = Ht[0][0] * x + Ht[0][1] * y + Ht[0][2];
      final y_ = Ht[1][0] * x + Ht[1][1] * y + Ht[1][2];
      final w_ = Ht[2][0] * x + Ht[2][1] * y + Ht[2][2];

      final warpedX = x_ / w_;
      final warpedY = y_ / w_;

      warpedPoints[i][j] = [warpedX, warpedY];
    }
  }

  return warpedPoints;
}


List<List<List<double>>> getIntersectionPoints(
    List<List<double>> horizontalLines, List<List<double>> verticalLines) {
  int numRows = horizontalLines.length;

  List<List<List<double>>> intersectionPoints = [];


  for (int j = 0; j < numRows; j++) {
    double rho = horizontalLines[j][0];
    double theta = horizontalLines[j][1];

    List<List<double>> points = getIntersectionPointsWithOneLine(verticalLines, rho, theta);
    intersectionPoints.add(points);
  }

  return intersectionPoints;
}

List<int> chooseFromRange(int length) {
  final random = math.Random();
  if (length < 2) {
    throw ArgumentError("Need at least 2 elements to choose from");
  }
  int first = random.nextInt(length);
  int second;
  do {
    second = random.nextInt(length);
  } while (second == first);

  return [first, second]..sort(); // Keep it ordered if needed
}
  
/// Eliminates similar lines by clustering them based on their intersection
/// points with a reference perpendicular line and selecting a representative
/// line from each cluster.
List<List<double>> eliminateSimilarLines(
    List<List<double>> lines, List<List<double>> perpendicularLines,
    {double epsilon = 12.0, int minPoints = 1}) {
  // Compute the average perpendicular line
  double perpRho = perpendicularLines.map((line) => line[0]).average;
  double perpTheta = perpendicularLines.map((line) => line[1]).average;

  // Compute intersection points
  List<List<double>> intersectionPoints =
      getIntersectionPointsWithOneLine(lines, perpRho, perpTheta);

  // Apply DBSCAN clustering
  DBSCAN dbscan = DBSCAN(epsilon: epsilon, minPoints: minPoints);
  List<List<int>> clusters = dbscan.run(intersectionPoints);
  List<int> labels = dbscan.label!;

  // Map to store cluster index and corresponding lines
  Map<int, List<List<double>>> clusterMap = {};

  for (int i = 0; i < labels.length; i++) {
    int label = labels[i];
    if (!clusterMap.containsKey(label)) {
      clusterMap[label] = [];
    }
    clusterMap[label]!.add(lines[i]);
  }

  // Select representative lines from each cluster
  List<List<double>> filteredLines = [];

  for (var cluster in clusterMap.values) {
    // Sort lines in the cluster based on rho
    cluster.sort((a, b) => a[0].compareTo(b[0]));
    // Select the median line
    filteredLines.add(cluster[cluster.length ~/ 2]);
  }

  return filteredLines;
}

  List<List<double>> getIntersectionPointsWithOneLine(
    List<List<double>> lines, double perpRho, double perpTheta) {
  List<List<double>> intersectionPoints = [];

  double cosPerpTheta = math.cos(perpTheta);
  double sinPerpTheta = math.sin(perpTheta);

  for (var line in lines) {
    double rho = line[0];
    double theta = line[1];

    double cosTheta = math.cos(theta);
    double sinTheta = math.sin(theta);

    double denominator = cosPerpTheta * sinTheta - cosTheta * sinPerpTheta;

    // To avoid division by zero
    if (denominator == 0) {
      continue;
    }

    double x = (sinTheta * perpRho - sinPerpTheta * rho) / denominator;
    double y = (cosTheta * perpRho - cosPerpTheta * rho) / -denominator;

    intersectionPoints.add([x, y]);
  }

  return intersectionPoints;
}

  /// Detect lines using Hough transform
  List<List<double>> detectLines(cv.Mat edges) {
    // Perform Hough line detection
    // Returns array of [rho, theta] pairs
    final lines = cv.HoughLines(
      edges,
      1.0,
      math.pi / 360,
      config.houghThreshold,
    );

    if (lines.isEmpty) {
      return [];
    }

    // Convert lines to list format and fix negative rho values
    var fixedLines = fixNegativeRhoInHesseNormalForm(lines);

    // Optional: Eliminate diagonal lines
    // Keep only vertical and horizontal lines within threshold
    final threshold = 45 * math.pi / 180; // 45 degrees in radians

    fixedLines =
        fixedLines.where((line) {
          final theta = line[1];
          // Check if line is vertical (close to 0 degrees)
          final isVertical = theta.abs() < threshold;
          // Check if line is horizontal (close to 90 degrees)
          final isHorizontal = (theta - math.pi / 2).abs() < threshold;
          return isVertical || isHorizontal;
        }).toList();

    return fixedLines;
  }

  /// Fix negative rho values in Hesse normal form
  List<List<double>> fixNegativeRhoInHesseNormalForm(cv.Mat lines) {
    final List<List<double>> result = [];
    for (var i = 0; i < lines.rows; i++) {
      final rho = lines.at<double>(i, 0);
      final theta = lines.at<double>(i, 1);

      if (rho < 0) {
        result.add([-rho, theta - math.pi]);
      } else {
        result.add([rho, theta]);
      }
    }
    return result;
  }

  /// Calculate absolute angle difference between two angles in radians
  double absoluteAngleDifference(List<double> x, List<double> y) {
    final diff = (x[1] - y[1]).abs() % (2 * math.pi);
    return math.min(diff, math.pi - diff);
  }

  /// Sort lines by rho values
  List<List<double>> sortLines(List<List<double>> lines) {
    if (lines.isEmpty) {
      return lines;
    }

    // Sort based on rho values (first element of each line)
    lines.sort((a, b) => a[0].compareTo(b[0]));
    return lines;
  }

  /// Cluster lines into horizontal and vertical groups
  Tuple2<List<List<double>>, List<List<double>>>
  clusterHorizontalAndVerticalLines(List<List<double>> lines) {
    if (lines.isEmpty) {
      return Tuple2([], []);
    }

    final sortedLines = sortLines(lines);

    // Use hierarchical clustering with single linkage
    final hierarchical = Hierarchical(
      minCluster: 2, // We want exactly 2 clusters (horizontal and vertical)
      linkage: LINKAGE.AVERAGE,
      distanceMeasure: absoluteAngleDifference,
    );

    hierarchical.run(sortedLines);

    final lineLabels = hierarchical.label;

    final angleWithYAxis = List<double>.empty(growable: true);
    for (var i = 0; i < sortedLines.length; i++) {
      final line = sortedLines[i];
      angleWithYAxis.add(absoluteAngleDifference(line, [0.0, 0.0]));
    }

    // Separate lines into horizontal and vertical based on their angles
    final List<List<double>> horizontalLines;
    final List<List<double>> verticalLines;

    final indicesWithLabel0 = lineLabels.where((label) => label == 0).toList();
    final indicesWithLabel1 = lineLabels.where((label) => label == 1).toList();

    double averageAngleWithYAxis0 = 0;
    for (var i in indicesWithLabel0) {
      averageAngleWithYAxis0 += angleWithYAxis[i];
    }
    averageAngleWithYAxis0 /= indicesWithLabel0.length;

    double averageAngleWithYAxis1 = 0;
    for (var i in indicesWithLabel1) {
      averageAngleWithYAxis1 += angleWithYAxis[i];
    }
    averageAngleWithYAxis1 /= indicesWithLabel1.length;

    if (averageAngleWithYAxis0 > averageAngleWithYAxis1) {
      horizontalLines = sortedLines.where((line) => lineLabels[sortedLines.indexOf(line)] == 0).toList();
      verticalLines = sortedLines.where((line) => lineLabels[sortedLines.indexOf(line)] == 1).toList();
    } else {
      horizontalLines = sortedLines.where((line) => lineLabels[sortedLines.indexOf(line)] == 1).toList();
      verticalLines = sortedLines.where((line) => lineLabels[sortedLines.indexOf(line)] == 0).toList();
    }

    return Tuple2(horizontalLines, verticalLines);
  }
}

double absoluteAngleDifference(List<double> point1, List<double> point2) {
  // double sum = 0;
  // int loop = min(point1.length, point2.length);

  // for (int i = 0; i < loop; i++) {
  //   sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
  // }

  // return sqrt(sum);
}

class ChessboardNotLocatedException implements Exception {
  final String message;
  ChessboardNotLocatedException(this.message);

  @override
  String toString() => 'ChessboardNotLocatedException: $message';
}

/// Extension method to wrap platform-specific code
extension CornerDetectionMethods on CornerDetector {
  /// This method would use platform channels to call native code for more accurate corner detection
  Future<Tuple3<List<List<double>>, List<List<double>>, List<int>>>
  findCornersNative(Uint8List imageBytes, int width, int height) async {
    // This would be implemented using platform channels to call native OpenCV functions
    // For now, convert the bytes to an img.Image and use the Dart implementation
    final image = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: imageBytes.buffer,
      numChannels: 4,
    );

    return findCorners(image);
  }
}
