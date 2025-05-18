import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set default tensor type to float32 (required for MPS)
torch.set_default_dtype(torch.float32)

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Constants
INPUT_SIZE = 300  # EfficientNet-B3 recommended input size
EMBEDDING_SIZE = 128  # Size of the embeddings

class PairDataset(Dataset):
    """Dataset for loading paired chess piece images."""
    
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image pairs.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.pairs_df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img1_path = self.pairs_df.iloc[idx, 0]
        img2_path = self.pairs_df.iloc[idx, 1]
        
        # Ensure we're using absolute paths if they're relative in the CSV
        if not os.path.isabs(img1_path):
            img1_path = os.path.abspath(img1_path)
        if not os.path.isabs(img2_path):
            img2_path = os.path.abspath(img2_path)
            
        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img2_path).convert('RGB')
        
        label = self.pairs_df.iloc[idx, 2]
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            
        # Explicitly convert label to float32 for MPS compatibility
        return image1, image2, torch.tensor(float(label), dtype=torch.float32)

class SiameseNetwork(nn.Module):
    """Siamese network with EfficientNet-B3 backbone."""
    
    def __init__(self, embedding_size=EMBEDDING_SIZE, pretrained=True):
        """
        Args:
            embedding_size (int): Size of the output embeddings.
            pretrained (bool): Whether to use pretrained weights.
        """
        super(SiameseNetwork, self).__init__()
        
        # Load pre-trained EfficientNet-B3 model
        if pretrained:
            self.efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)
        
        # Freeze most of the network
        for name, param in self.efficientnet.named_parameters():
            # Only keep the last few layers trainable
            print(name);
            print(param.shape);
            if 'features.8' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Get the number of features in the last layer
        num_features = self.efficientnet.classifier[1].in_features

        print(num_features);
        
        # Remove the original classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Create new embedding layers
        self.embedding = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )
        
    def forward_one(self, x):
        """Forward pass for one input."""
        x = self.efficientnet(x)
        x = self.embedding(x)
        return x
    
    def forward(self, input1, input2):
        """Forward pass for Siamese network."""
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese network.
    
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        Args:
            output1: First embedding
            output2: Second embedding
            label: 1 if same class, 0 if different class
        """
        # Euclidean distance between embeddings
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Contrastive loss formula
        loss_contrastive = torch.mean(
            (1-label) * torch.pow(euclidean_distance, 2) + 
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs=25, 
    eval_every=5,
    scheduler=None
):
    """
    Trains the model.
    
    Args:
        model: The Siamese network model
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        eval_every: Evaluate every n epochs
        scheduler: Learning rate scheduler
        
    Returns:
        Trained model and training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed, Loss: {epoch_loss:.4f}')
        
        # Step the scheduler if provided
        if scheduler:
            scheduler.step()
        
        # Evaluate on validation set every few epochs
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            for key, value in val_metrics.items():
                history[key].append(value)
            
            print(f'Validation - Loss: {val_metrics["val_loss"]:.4f}, '
                  f'Accuracy: {val_metrics["val_accuracy"]:.4f}, '
                  f'F1: {val_metrics["val_f1"]:.4f}')
            
            # Early stopping could be added here
    
    return model, history

def evaluate_model(model, data_loader, criterion, device, threshold=0.5):
    """
    Evaluates the model on a dataset.
    
    Args:
        model: The Siamese network model
        data_loader: DataLoader for dataset
        criterion: Loss function
        device: Device to evaluate on
        threshold: Distance threshold for classification
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, label in data_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            running_loss += loss.item()
            
            # Calculate distance
            distance = F.pairwise_distance(output1, output2)
            
            # Predict using threshold on distance
            # If distance < threshold, predict same class (1), else different class (0)
            predictions = (distance < threshold).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1
    }

def plot_training_history(history, save_path=None):
    """
    Plots the training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot, if provided
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(range(0, len(history['train_loss']), len(history['train_loss'])//len(history['val_loss'])), 
                 history['val_loss'], 'o-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot validation metrics
    plt.subplot(2, 1, 2)
    if 'val_accuracy' in history and history['val_accuracy']:
        plt.plot(history['val_accuracy'], 'o-', label='Accuracy')
    if 'val_precision' in history and history['val_precision']:
        plt.plot(history['val_precision'], 'o-', label='Precision')
    if 'val_recall' in history and history['val_recall']:
        plt.plot(history['val_recall'], 'o-', label='Recall')
    if 'val_f1' in history and history['val_f1']:
        plt.plot(history['val_f1'], 'o-', label='F1 Score')
    plt.title('Validation Metrics')
    plt.ylabel('Score')
    plt.xlabel('Evaluation Steps')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_checkpoint(model, optimizer, history, filename):
    """
    Saves a checkpoint of the model.
    
    Args:
        model: The model to save
        optimizer: The optimizer used
        history: Training history
        filename: Path to save the checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

def load_checkpoint(model, optimizer, filename, device):
    """
    Loads a checkpoint.
    
    Args:
        model: The model architecture
        optimizer: The optimizer
        filename: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded model, optimizer, and history
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    history = checkpoint.get('history', {})
    return model, optimizer, history

def main():
    parser = argparse.ArgumentParser(
        description="Train a Siamese neural network for chess piece similarity recognition"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the dataset directory containing pairs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train (default: 30)")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for contrastive loss (default: 1.0)")
    parser.add_argument("--embedding_size", type=int, default=EMBEDDING_SIZE,
                        help=f"Size of embeddings (default: {EMBEDDING_SIZE})")
    parser.add_argument("--save_model", action="store_true",
                        help="Save the trained model")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save models (default: 'models')")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--use_mps", action="store_true", default=True,
                        help="Use MPS (Metal Performance Shaders) for Apple Silicon acceleration")
    args = parser.parse_args()
    
    # Check if dataset directory exists
    pairs_dir = os.path.join(args.dataset_dir, "pairs")
    if not os.path.exists(pairs_dir):
        print(f"Dataset directory not found: {pairs_dir}")
        return
    
    # Create model directory if it doesn't exist
    if args.save_model and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # Set device - with support for Apple Silicon (M3 Pro in this case)
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders) device for Neural Engine acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Define data transforms for EfficientNet-B3
    data_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_csv = os.path.join(pairs_dir, "train_pairs.csv")
    val_csv = os.path.join(pairs_dir, "val_pairs.csv")
    test_csv = os.path.join(pairs_dir, "test_pairs.csv")
    
    train_dataset = PairDataset(train_csv, transform=data_transforms)
    val_dataset = PairDataset(val_csv, transform=data_transforms)
    test_dataset = PairDataset(test_csv, transform=data_transforms)
    
    # Determine num_workers based on system - reduce for MPS to avoid issues
    num_workers = 0 if device.type == "mps" else 4
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model, criterion, and optimizer
    model = SiameseNetwork(embedding_size=args.embedding_size, pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    history = {}
    
    # Load checkpoint if provided
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading checkpoint from {args.load_checkpoint}")
        model, optimizer, history = load_checkpoint(model, optimizer, args.load_checkpoint, device)
    
    # Train model
    print("Starting training...")
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device,
        num_epochs=args.epochs,
        eval_every=5,
        scheduler=scheduler
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print(f"Test metrics - Loss: {test_metrics['val_loss']:.4f}, "
          f"Accuracy: {test_metrics['val_accuracy']:.4f}, "
          f"F1: {test_metrics['val_f1']:.4f}")
    
    # Plot training history
    plot_dir = os.path.join(args.model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plot_training_history(history, save_path=plot_path)
    
    # Save the model
    if args.save_model:
        model_path = os.path.join(args.model_dir, f"siamese_efficientnet_b3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        save_checkpoint(model, optimizer, history, model_path)

if __name__ == "__main__":
    main()
