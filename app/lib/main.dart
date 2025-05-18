import 'package:flutter/material.dart';
import 'package:chess/chess.dart' as chess;
import 'chess_recorder/index.dart';

void main() {
  // Ensure Flutter is initialized
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ChessRecorderApp());
}

class ChessRecorderApp extends StatelessWidget {
  const ChessRecorderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Chess Game Recorder',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const ChessRecorderScreen(),
    );
  }
}

class ChessRecorderScreen extends StatefulWidget {
  const ChessRecorderScreen({super.key});

  @override
  State<ChessRecorderScreen> createState() => _ChessRecorderScreenState();
}

class _ChessRecorderScreenState extends State<ChessRecorderScreen> {
  final List<chess.Move> _moves = [];
  bool _boardFound = false;
  bool _gameInProgress = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Chess Game Recorder'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: _showInfo,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            flex: 3,
            child: ChessRecorderView(
              config: const ChessRecorderConfig(
                debugMode: true,
                pieceConfidenceThreshold: 0.5,
                minMoveDetections: 3,
                moveBufferWindowMs: 500,
              ),
              onBoardFound: _handleBoardFound,
              onGameStarted: _handleGameStarted,
              onMoveDetected: _handleMoveDetected,
              onError: _handleError,
            ),
          ),
          Expanded(flex: 1, child: _buildMovesList()),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _resetGame,
        tooltip: 'Reset Game',
        child: const Icon(Icons.refresh),
      ),
    );
  }

  Widget _buildMovesList() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Moves',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                _buildGameStatus(),
              ],
            ),
          ),
          Expanded(
            child:
                _moves.isEmpty
                    ? const Center(child: Text('No moves recorded yet'))
                    : ListView.builder(
                      itemCount: _moves.length,
                      itemBuilder: (context, index) {
                        final move = _moves[index];
                        final moveNumber = (index ~/ 2) + 1;
                        final isWhite = index % 2 == 0;

                        if (isWhite) {
                          return ListTile(
                            dense: true,
                            leading: Text(
                              '$moveNumber.',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            title: Text(move.toString()),
                          );
                        } else {
                          return ListTile(
                            dense: true,
                            leading: const Text(''),
                            title: Text(move.toString()),
                          );
                        }
                      },
                    ),
          ),
        ],
      ),
    );
  }

  Widget _buildGameStatus() {
    Color statusColor;
    String statusText;

    if (_gameInProgress) {
      statusColor = Colors.green;
      statusText = 'Game in progress';
    } else if (_boardFound) {
      statusColor = Colors.orange;
      statusText = 'Board found';
    } else {
      statusColor = Colors.grey;
      statusText = 'Searching';
    }

    return Row(
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(color: statusColor, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text(statusText),
      ],
    );
  }

  void _handleBoardFound() {
    setState(() {
      _boardFound = true;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Chess board detected!'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  void _handleGameStarted() {
    setState(() {
      _gameInProgress = true;
    });
  }

  void _handleMoveDetected(chess.Move move) {
    setState(() {
      _moves.add(move);
    });
  }

  void _handleError(String error) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Error: $error'), backgroundColor: Colors.red),
    );
  }

  void _resetGame() {
    setState(() {
      _moves.clear();
      _boardFound = false;
      _gameInProgress = false;
    });
  }

  void _showInfo() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Chess Recorder'),
            content: const Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('This app records chess games from your camera.'),
                SizedBox(height: 8),
                Text('Instructions:'),
                Text('1. Point the camera at a chess board'),
                Text('2. Keep the board in frame during the game'),
                Text('3. The app will record all moves automatically'),
                Text('4. Tap the reset button to start a new game'),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('OK'),
              ),
            ],
          ),
    );
  }
}
