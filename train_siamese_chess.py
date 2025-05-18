"""
Chess Piece Siamese Network Training and Inference

This script trains a Siamese Neural Network to compare chess pieces and determine if they are of the same type.
It uses pre-generated pairs of chess piece images stored in CSV files.

Usage examples:

1. Training from scratch:
   python train_siamese_chess.py --mode train --dataset_path /path/to/dataset --epochs 20 --batch_size 64

2. Resume training from checkpoint:
   python train_siamese_chess.py --mode train --dataset_path /path/to/dataset --checkpoint models/siamese_chess_checkpoint_epoch_5.pth

3. Run inference on test samples:
   python train_siamese_chess.py --mode inference --test_samples 10

4. Run inference on specific images:
   python train_siamese_chess.py --mode inference --img1 path/to/queen1.jpg --img2 path/to/queen2.jpg

5. Use a specific model for inference:
   python train_siamese_chess.py --mode inference --model_path models/siamese_chess_model_best.pth --img1 path/to/piece1.jpg --img2 path/to/piece2.jpg

6. Export a model for inference (removes training-specific data):
   python train_siamese_chess.py --mode export --model_path models/siamese_chess_model_best.pth --output_path models/inference_model.pth

All models are saved in a consistent format as dictionaries containing:
- model_state_dict: The model parameters
- optimizer_state_dict: (training checkpoints only) The optimizer state
- epoch: The epoch number (for checkpoints)
- val_accuracy: Validation accuracy at the time of saving
- best_val_accuracy: The best validation accuracy seen so far
- Other metadata flags like is_best, is_final, or is_inference_model
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from chess_piece_siamese_net import SiameseNetwork, SiameseContrastiveLoss, get_transforms

class ChessPiecePairDataset(Dataset):
    """Dataset for loading pairs of chess piece images from pre-generated CSV files"""
    def __init__(self, dataset_dir, split='train', transform=None):
        """
        Args:
            dataset_dir (str): Path to the dataset directory containing 'pairs' folder with CSV files
            split (str): One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on the images
        """
        self.transform = transform
        
        # Load the appropriate CSV file
        csv_path = os.path.join(dataset_dir, 'pairs', f'{split}_pairs.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Pairs file not found: {csv_path}")
        
        # Read the CSV file
        self.pairs_df = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.pairs_df)} pairs from {csv_path}")
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        img1_path = self.pairs_df.iloc[idx, 0]
        img2_path = self.pairs_df.iloc[idx, 1]
        label = self.pairs_df.iloc[idx, 2]
        
        # Adjust paths if needed (make absolute if relative)
        if not os.path.isabs(img1_path):
            img1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img1_path)
        if not os.path.isabs(img2_path):
            img2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img2_path)
        
        # Load and convert images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float)

def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    Evaluate a Siamese Network model on a dataset
    
    Args:
        model (SiameseNetwork): The model to evaluate
        data_loader (DataLoader): DataLoader containing the evaluation dataset
        device (torch.device): Device to run inference on
        threshold (float): Similarity threshold for binary classification
        
    Returns:
        dict: Dictionary containing evaluation metrics:
            - accuracy: Overall accuracy percentage
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - true_positives: Number of true positives
            - false_positives: Number of false positives
            - false_negatives: Number of false negatives
            - true_negatives: Number of true negatives
            - total: Total number of samples evaluated
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    with torch.no_grad():
        for img1, img2, label in data_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            similarity = model(img1, img2)
            pred = (similarity > threshold).float()
            total += label.size(0)

            # Squeeze pred to match label's shape
            pred = pred.squeeze(1)
            
            # Calculate number of correct predictions
            correct += (pred == label).sum().item()
            
            # Calculate confusion matrix elements
            true_positives += ((pred == 1) & (label == 1)).sum().item()
            false_positives += ((pred == 1) & (label == 0)).sum().item()
            false_negatives += ((pred == 0) & (label == 1)).sum().item()
            true_negatives += ((pred == 0) & (label == 0)).sum().item()
    
    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'total': total
    }

def print_metrics(metrics, prefix="Evaluation"):
    """
    Print evaluation metrics in a formatted way
    
    Args:
        metrics (dict): Dictionary of metrics from evaluate_model
        prefix (str): Prefix for the output (e.g., "Validation", "Test")
    """
    print(f'{prefix} Results on {metrics["total"]} samples:')
    print(f'  Accuracy: {metrics["accuracy"]:.2f}%')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall: {metrics["recall"]:.4f}')
    print(f'  F1 Score: {metrics["f1_score"]:.4f}')
    print(f'  Confusion Matrix:')
    print(f'    True Positives: {metrics["true_positives"]}')
    print(f'    False Positives: {metrics["false_positives"]}')
    print(f'    False Negatives: {metrics["false_negatives"]}')
    print(f'    True Negatives: {metrics["true_negatives"]}')
    

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=10, start_epoch=0, best_val_acc=0.0):
    model.train()
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for i, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Get model outputs - similarity probability
            similarity = model(img1, img2)
            
            # Calculate loss using the cross-entropy loss with regularization
            loss = criterion(similarity, label, model)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        
        # Evaluate on validation set after each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Print validation metrics
        print(f'Epoch {epoch+1} completed:')
        print(f'  Validation Accuracy: {val_metrics["accuracy"]:.2f}%')
        print(f'  Precision: {val_metrics["precision"]:.4f}')
        print(f'  Recall: {val_metrics["recall"]:.4f}')
        print(f'  F1 Score: {val_metrics["f1_score"]:.4f}')
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_metrics['accuracy'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1_score': val_metrics['f1_score'],
            'best_val_accuracy': best_val_acc
        }
        
        # Save checkpoint after each epoch
        torch.save(checkpoint, os.path.join('models', f'siamese_chess_checkpoint_epoch_{epoch+1}.pth'))
        
        # Save the best model based on validation accuracy
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            # Save with the same format as checkpoint for consistency
            checkpoint['is_best'] = True
            torch.save(checkpoint, os.path.join('models', 'siamese_chess_model_best.pth'))
            print(f'New best model saved with validation accuracy: {val_metrics["accuracy"]:.2f}%')
    
    # Save the final model in the same format
    final_checkpoint = {
        'epoch': start_epoch + num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_metrics['accuracy'],
        'precision': val_metrics['precision'],
        'recall': val_metrics['recall'],
        'f1_score': val_metrics['f1_score'],
        'best_val_accuracy': best_val_acc,
        'is_final': True
    }
    torch.save(final_checkpoint, os.path.join('models', 'siamese_chess_model_last.pth'))
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    
    # Evaluate on test set
    print("\nEvaluating final model on test set:")
    print("-----------------------------------")
    
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, "Test")
    
    # Update the final checkpoint with test metrics
    final_checkpoint['test_accuracy'] = test_metrics['accuracy']
    final_checkpoint['test_precision'] = test_metrics['precision']
    final_checkpoint['test_recall'] = test_metrics['recall']
    final_checkpoint['test_f1_score'] = test_metrics['f1_score']
    
    # Save the updated final model with test metrics
    torch.save(final_checkpoint, os.path.join('models', 'siamese_chess_model_last.pth'))
    
    # Also update the best model if it exists with test metrics
    best_model_path = os.path.join('models', 'siamese_chess_model_best.pth')
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        # Load the best model
        best_model = SiameseNetwork().to(device)
        best_model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Evaluate the best model on test set
        print("\nEvaluating best model on test set:")
        print("----------------------------------")
        best_test_metrics = evaluate_model(best_model, test_loader, device)
        print_metrics(best_test_metrics, "Best model test")
        
        # Update the best checkpoint with test metrics
        best_checkpoint['test_accuracy'] = best_test_metrics['accuracy']
        best_checkpoint['test_precision'] = best_test_metrics['precision']
        best_checkpoint['test_recall'] = best_test_metrics['recall']
        best_checkpoint['test_f1_score'] = best_test_metrics['f1_score']
        
        # Save the updated best model with test metrics
        torch.save(best_checkpoint, best_model_path)
    
    return model

def compare_pieces(model, img1_path, img2_path, transform=None, device=None, threshold=0.5):
    """
    Compare two chess piece images and determine if they are the same type
    
    Args:
        model (SiameseNetwork): Trained Siamese Network model
        img1_path (str): Path to the first image
        img2_path (str): Path to the second image
        transform: Optional transform to be applied to the images (will use default if None)
        device (torch.device): Device to run inference on (will use cuda if available if None)
        threshold (float): Similarity threshold (default: 0.5)
        
    Returns:
        is_same_type (bool): Whether the two pieces are of the same type
        similarity_score (float): Raw similarity score from the model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if transform is None:
        transform = get_transforms(img_size=105)
    
    # Load and preprocess images
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    if transform:
        img1 = transform(img1)
        img2 = transform(img2)
    
    # Add batch dimension
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get similarity score
    with torch.no_grad():
        similarity = model(img1, img2).item()
    
    return similarity > threshold, similarity

def load_model(model_path, device=None):
    """
    Load a trained Siamese Network model
    
    Args:
        model_path (str): Path to the saved model file
        device (torch.device): Device to load the model on (will use cuda if available if None)
        
    Returns:
        model (SiameseNetwork): Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved file    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize the model
    model = SiameseNetwork().to(device)
    
    # Check if the loaded object is a state_dict directly or a checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # It's a checkpoint dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        if 'val_accuracy' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_accuracy']:.2f}%")
        if 'precision' in checkpoint:
            print(f"Model precision: {checkpoint['precision']:.4f}")
        if 'recall' in checkpoint:
            print(f"Model recall: {checkpoint['recall']:.4f}")
        if 'f1_score' in checkpoint:
            print(f"Model F1 score: {checkpoint['f1_score']:.4f}")
        if checkpoint.get('is_best', False):
            print("This is the best performing model from training.")
        if checkpoint.get('is_final', False):
            print("This is the final model from training.")
    
    model.eval()
    return model

def export_inference_model(checkpoint_path, output_path, device=None):
    """
    Export an inference-only model from a checkpoint
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        output_path (str): Path where to save the inference model
        device (torch.device): Device to load the model on
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the model
        model = load_model(checkpoint_path, device)
        
        # Save just the state_dict for inference
        inference_model = {
            'model_state_dict': model.state_dict(),
            'is_inference_model': True,
            'export_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get original metadata if available
        checkpoint = torch.load(checkpoint_path, map_location=device if device else "cpu")
        if isinstance(checkpoint, dict):
            # Copy validation metrics
            for metric in ['val_accuracy', 'precision', 'recall', 'f1_score', 'best_val_accuracy']:
                if metric in checkpoint:
                    inference_model[metric] = checkpoint[metric]
            
            # Copy test metrics if present
            for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']:
                if metric in checkpoint:
                    inference_model[metric] = checkpoint[metric]
            
            # Copy other metadata
            if 'epoch' in checkpoint:
                inference_model['source_epoch'] = checkpoint['epoch']
            if checkpoint.get('is_best', False):
                inference_model['is_from_best'] = True
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save the inference model
        torch.save(inference_model, output_path)
        print(f"Exported inference-only model to {output_path}")
        
        # Print included metrics
        print("\nModel metrics included in export:")
        if 'val_accuracy' in inference_model:
            print(f"  Validation Accuracy: {inference_model['val_accuracy']:.2f}%")
        if 'precision' in inference_model:
            print(f"  Precision: {inference_model['precision']:.4f}")
        if 'recall' in inference_model:
            print(f"  Recall: {inference_model['recall']:.4f}")
        if 'f1_score' in inference_model:
            print(f"  F1 Score: {inference_model['f1_score']:.4f}")
        if 'test_accuracy' in inference_model:
            print(f"  Test Accuracy: {inference_model['test_accuracy']:.2f}%")
        if 'test_precision' in inference_model:
            print(f"  Test Precision: {inference_model['test_precision']:.4f}")
        if 'test_recall' in inference_model:
            print(f"  Test Recall: {inference_model['test_recall']:.4f}")
        if 'test_f1_score' in inference_model:
            print(f"  Test F1 Score: {inference_model['test_f1_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"Error exporting inference model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train or run inference with a Siamese Network for chess piece type comparison')
    
    # Common arguments
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'export'], required=True,
                        help='Mode to run: "train", "inference", or "export"')
    parser.add_argument('--dataset_path', type=str, default='/Users/sahilchaddha/projects/chess_recorder/dataset/chess_piece_similarity',
                        help='Path to the dataset directory')
    parser.add_argument('--img_size', type=int, default=105,
                        help='Size of the images for the network (default: 105)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: None, will use cuda if available)')
    
    # Training arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--lambda_reg', type=float, default=0.0001,
                        help='L2 regularization coefficient (default: 0.0001)')
    
    # Inference arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model for inference')
    parser.add_argument('--test_samples', type=int, default=5,
                        help='Number of test samples to evaluate in inference mode (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold for inference (default: 0.5)')
    parser.add_argument('--img1', type=str, default=None,
                        help='Path to first image for custom comparison')
    parser.add_argument('--img2', type=str, default=None,
                        help='Path to second image for custom comparison')
    
    # Export arguments
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path where to save the exported inference model')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Get transforms
    transform = get_transforms(img_size=args.img_size)
    
    if args.mode == 'train':
        # Training mode
        print(f"Training mode selected. Dataset path: {args.dataset_path}")
        
        # Load datasets
        train_dataset = ChessPiecePairDataset(args.dataset_path, split='train', transform=transform)
        val_dataset = ChessPiecePairDataset(args.dataset_path, split='val', transform=transform)
        test_dataset = ChessPiecePairDataset(args.dataset_path, split='test', transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = SiameseNetwork(input_channels=1, embedding_dim=4096).to(device)
        
        # Set up optimizer and criterion
        criterion = SiameseContrastiveLoss(lambda_reg=args.lambda_reg)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Load from checkpoint if provided
        start_epoch = 0
        best_val_acc = 0.0
        if args.checkpoint:
            if os.path.exists(args.checkpoint):
                print(f"Loading checkpoint from {args.checkpoint}")
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_acc = checkpoint.get('best_val_accuracy', 0.0)
                print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.2f}%")
            else:
                print(f"Warning: Checkpoint file {args.checkpoint} not found. Starting from scratch.")
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Train model (adjust num_epochs to account for starting epoch)
        remaining_epochs = args.epochs - start_epoch
        if remaining_epochs <= 0:
            print("Warning: Start epoch >= total epochs. No training will be performed.")
            return
            
        train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, 
              num_epochs=remaining_epochs, start_epoch=start_epoch, best_val_acc=best_val_acc)
        
        print("Training completed!")
        
    elif args.mode == 'inference':
        # Inference mode
        print(f"Inference mode selected. Dataset path: {args.dataset_path}")
        
        # Determine model path
        model_path = args.model_path
        if not model_path:
            # Try to use best model from models directory
            default_model_path = os.path.join('models', 'siamese_chess_model_best.pth')
            if os.path.exists(default_model_path):
                model_path = default_model_path
                print(f"Using default model at {model_path}")
            else:
                print("Error: No model path provided and default model not found.")
                return
        elif not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found.")
            return
        
        # Load model
        inference_model = load_model(model_path, device)
        
        # Check if custom image comparison is requested
        if args.img1 and args.img2:
            print("\nComparing custom images:")
            print("------------------------")
            print(f"Image 1: {args.img1}")
            print(f"Image 2: {args.img2}")
            
            # Verify files exist
            if not os.path.exists(args.img1):
                print(f"Error: Image 1 file {args.img1} not found.")
                return
            if not os.path.exists(args.img2):
                print(f"Error: Image 2 file {args.img2} not found.")
                return
                
            # Compare images
            is_same, similarity = compare_pieces(inference_model, args.img1, args.img2, transform, device, threshold=args.threshold)
            
            print(f"Prediction: {'Same type' if is_same else 'Different type'}")
            print(f"Similarity score: {similarity:.4f}")
            
        else:
            # Load test dataset for standard evaluation
            test_dataset = ChessPiecePairDataset(args.dataset_path, split='test', transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Run inference on test samples
            print("\nRunning inference on test samples:")
            print("--------------------------------")
            
            # First evaluate specific samples if requested
            if args.test_samples > 0:
                correct = 0
                total = 0
                
                # Ensure we don't try to evaluate more samples than exist in the dataset
                num_samples = min(args.test_samples, len(test_dataset))
                
                for i in range(num_samples):
                    img1, img2, true_label = test_dataset[i]
                    
                    # Get paths from dataset
                    img1_path = test_dataset.pairs_df.iloc[i, 0]
                    img2_path = test_dataset.pairs_df.iloc[i, 1]
                    
                    # Adjust paths if needed
                    if not os.path.isabs(img1_path):
                        img1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), img1_path)
                    if not os.path.isabs(img2_path):
                        img2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), img2_path)
                    
                    # Compare the pieces
                    is_same, similarity = compare_pieces(inference_model, img1_path, img2_path, transform, device, threshold=args.threshold)
                    
                    # Update accuracy statistics
                    total += 1
                    if (is_same and true_label.item() == 1) or (not is_same and true_label.item() == 0):
                        correct += 1
                    
                    print(f"Example {i+1}:")
                    print(f"  Image 1: {os.path.basename(img1_path)}")
                    print(f"  Image 2: {os.path.basename(img2_path)}")
                    print(f"  True label: {'Same type' if true_label.item() == 1 else 'Different type'}")
                    print(f"  Prediction: {'Same type' if is_same else 'Different type'}")
                    print(f"  Similarity score: {similarity:.4f}")
                    print(f"  Correct: {'Yes' if (is_same and true_label.item() == 1) or (not is_same and true_label.item() == 0) else 'No'}")
                    print("")
                
                # Print overall accuracy
                if total > 0:
                    print(f"Sample accuracy on {total} test samples: {100 * correct / total:.2f}%")
            
            # Now evaluate on entire test set
            print("\nEvaluating on entire test set:")
            print("-----------------------------")
            
            # Use our new evaluate_model function
            test_metrics = evaluate_model(inference_model, test_loader, device, threshold=args.threshold)
            print_metrics(test_metrics, "Test")
    
    elif args.mode == 'export':
        # Export mode - create inference-only model
        print("Export mode selected. Creating inference-only model.")
        
        # Check if model path is provided
        if not args.model_path:
            # Try to use best model from models directory
            default_model_path = os.path.join('models', 'siamese_chess_model_best.pth')
            if os.path.exists(default_model_path):
                args.model_path = default_model_path
                print(f"Using default model at {args.model_path}")
            else:
                print("Error: No model path provided and default model not found.")
                return
        elif not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found.")
            return
        
        # Check if output path is provided
        if not args.output_path:
            # Use a default output path
            model_name = os.path.splitext(os.path.basename(args.model_path))[0]
            args.output_path = os.path.join('models', f"{model_name}_inference.pth")
            print(f"No output path provided. Using default: {args.output_path}")
        
        # Export the model
        export_inference_model(args.model_path, args.output_path, device)
    
    else:
        print(f"Error: Invalid mode {args.mode}")

if __name__ == "__main__":
    main()