import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:provider/provider.dart';
import 'package:chess/chess.dart' as chess;

import 'chess_recorder.dart';
import 'chess_models.dart';

/// Widget that displays the chess recorder with camera preview and board visualization
class ChessRecorderView extends StatefulWidget {
  /// Configuration options for the chess recorder
  final ChessRecorderConfig config;

  /// Callback when a move is detected
  final void Function(chess.Move)? onMoveDetected;

  /// Callback when board is found
  final void Function()? onBoardFound;

  /// Callback when game starts
  final void Function()? onGameStarted;

  /// Callback when an error occurs
  final void Function(String)? onError;

  const ChessRecorderView({
    super.key,
    this.config = const ChessRecorderConfig(),
    this.onMoveDetected,
    this.onBoardFound,
    this.onGameStarted,
    this.onError,
  });

  @override
  State<ChessRecorderView> createState() => _ChessRecorderViewState();
}

class _ChessRecorderViewState extends State<ChessRecorderView> {
  late ChessRecorder _recorder;
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;

  bool _isCameraInitialized = false;
  RecorderState _recorderState = RecorderState.initialized;
  ChessboardState? _latestBoardState;
  String _currentFen = '';

  Timer? _processingTimer;

  @override
  void initState() {
    super.initState();
    _initRecorder();
    _initCamera();
  }

  Future<void> _initRecorder() async {
    _recorder = ChessRecorder(
      config: widget.config,
      onStateChanged: _handleStateChanged,
      onBoardDetected: _handleBoardDetected,
      onMoveDetected: _handleMoveDetected,
      onError: _handleError,
      fps: 10,
    );

    await _recorder.initialize();
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras != null && _cameras!.isNotEmpty) {
        // Select the back camera by default
        final camera = _cameras!.firstWhere(
          (cam) => cam.lensDirection == CameraLensDirection.back,
          orElse: () => _cameras!.first,
        );

        await _setupCamera(camera);
      } else {
        _handleError('No cameras available');
      }
    } catch (e) {
      _handleError('Error initializing camera: $e');
    }
  }

  Future<void> _setupCamera(CameraDescription camera) async {
    if (_cameraController != null) {
      await _cameraController!.dispose();
    }

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      fps: 10, // Balance between quality and performance
      enableAudio: false,
      imageFormatGroup:
          Platform.isAndroid
              ? ImageFormatGroup.yuv420
              : ImageFormatGroup.bgra8888,
    );

    try {
      await _cameraController!.initialize();

      // Start image stream
      await _cameraController!.startImageStream(_processImageStream);

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      _handleError('Error setting up camera: $e');
    }
  }

  void _processImageStream(CameraImage image) async {
    // Limit processing rate
    if (_processingTimer != null && _processingTimer!.isActive) return;

    _processingTimer = Timer(const Duration(milliseconds: 100), () {});

    try {
      // Convert CameraImage to img.Image
      final processedImage = await _convertCameraImage(image);
      if (processedImage == null) return;

      // Get current time in seconds
      final frameTime = DateTime.now().millisecondsSinceEpoch / 1000.0;

      // Process frame
      final boardState = await _recorder.processFrame(
        processedImage,
        frameTime,
      );

      // Update UI
      if (boardState != null && mounted) {
        setState(() {
          _latestBoardState = boardState;
          _currentFen = _recorder.getCurrentFen();
        });
      }
    } catch (e) {
      // Just log processing errors without failing
      print('Error processing image: $e');
    }
  }

  Future<cv.Mat?> _convertCameraImage(CameraImage image) async {
    try {
      if (Platform.isAndroid) {
        // Convert YUV420 to RGB
        return _convertYUV420toImage(image);
      } else {
        // Convert BGRA to RGB
        return _convertBGRA8888toImage(image);
      }
    } catch (e) {
      print('Error converting image: $e');
      return null;
    }
  }

  cv.Mat _convertYUV420toImage(CameraImage image) {
    // Convert image to Uint8List (Assuming NV21 format)
    Uint8List uint8List = image.planes[0].bytes;
    // Convert to OpenCV Mat
    cv.Mat yuvMat = cv.Mat.fromList(
      image.height,
      image.width,
      cv.MatType.CV_8UC1,
      uint8List,
    );

    final rgbMat = cv.cvtColor(yuvMat, cv.COLOR_YUV2RGB_NV21);

    return rgbMat;
  }

  cv.Mat _convertBGRA8888toImage(CameraImage image) {
    // Create a new image with the right dimensions
    final imgWidth = image.width;
    final imgHeight = image.height;

    final bgraMat = cv.Mat.fromList(
      imgHeight,
      imgWidth,
      cv.MatType.CV_8UC4,
      image.planes[0].bytes,
    );

    return cv.cvtColor(bgraMat, cv.COLOR_BGRA2RGB);
  }

  void _handleStateChanged(RecorderState state) {
    if (mounted) {
      setState(() {
        _recorderState = state;
      });

      if (state == RecorderState.boardFound && widget.onBoardFound != null) {
        widget.onBoardFound!();
      } else if (state == RecorderState.gameInProgress &&
          widget.onGameStarted != null) {
        widget.onGameStarted!();
      }
    }
  }

  void _handleBoardDetected(ChessboardState boardState) {
    if (mounted) {
      setState(() {
        _latestBoardState = boardState;
      });
    }
  }

  void _handleMoveDetected(chess.Move move) {
    if (widget.onMoveDetected != null) {
      widget.onMoveDetected!(move);
    }

    if (mounted) {
      setState(() {
        _currentFen = _recorder.getCurrentFen();
      });
    }
  }

  void _handleError(String error) {
    print('Chess recorder error: $error');
    if (widget.onError != null) {
      widget.onError!(error);
    }
  }

  @override
  void dispose() {
    _processingTimer?.cancel();
    _cameraController?.dispose();
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(flex: 3, child: _buildCameraPreview()),
        Expanded(flex: 2, child: _buildChessboardView()),
        if (widget.config.debugMode)
          Expanded(flex: 1, child: _buildDebugInfo()),
      ],
    );
  }

  Widget _buildCameraPreview() {
    if (!_isCameraInitialized || _cameraController == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return ClipRect(
      child: AspectRatio(
        aspectRatio: _cameraController!.value.aspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(_cameraController!),

            // Overlay state information
            Positioned(
              top: 20,
              left: 20,
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'State: ${_recorderState.toString().split('.').last}',
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChessboardView() {
    return Container(
      padding: const EdgeInsets.all(8),
      color: Colors.grey[200],
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Chessboard',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: Center(
              child:
                  _latestBoardState != null &&
                          _latestBoardState!.warpedImage != null
                      ? Image.memory(
                        _convertImageToBytes(_latestBoardState!.warpedImage!),
                        fit: BoxFit.contain,
                      )
                      : const Text('Searching for chessboard...'),
            ),
          ),
          const SizedBox(height: 8),
          Text('FEN: $_currentFen'),
        ],
      ),
    );
  }

  Widget _buildDebugInfo() {
    return Container(
      padding: const EdgeInsets.all(8),
      color: Colors.grey[300],
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Debug Info',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Expanded(
            child: ListView(
              children: [
                Text(
                  'Board quality: ${_latestBoardState?.qualityScore.toStringAsFixed(2) ?? "N/A"}',
                ),
                Text(
                  'Detected pieces: ${_latestBoardState?.pieces.length ?? 0}',
                ),
                if (_recorder.lastCommittedMove != null)
                  Text('Last move: ${_recorder.lastCommittedMove!.toString()}'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Uint8List _convertImageToBytes(img.Image image) {
    // Convert img.Image to PNG bytes
    return Uint8List.fromList(img.encodePng(image));
  }
}
