import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from chess_piece_siamese_net import get_transforms
from keras.utils import to_categorical

class ChessPieceDataset(Dataset):
    """Dataset for loading chess piece images with their class labels"""
    def __init__(self, dataset_dir, split='train', transform=None):
        """
        Args:
            dataset_dir (str): Path to the dataset directory
            split (str): One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on the images
        """
        self.transform = transform
        
        # Load the appropriate CSV file with image paths and class labels
        csv_path = os.path.join(dataset_dir, f'{split}_pieces.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Pieces file not found: {csv_path}")
        
        # Read the CSV file (assumes format: image_path, class_label)
        self.pieces_df = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.pieces_df)} pieces from {csv_path}")
    
    def __len__(self):
        return len(self.pieces_df)
    
    def __getitem__(self, idx):
        img_path = self.pieces_df.iloc[idx, 0]
        class_label = self.pieces_df.iloc[idx, 1]
        
        # Adjust path if needed (make absolute if relative)
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path)
        
        # Load and convert image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
            # Convert from (C, H, W) format to (H, W, C) format for Keras
            img = img.permute(1, 2, 0)
            
            # Convert image to numpy array
            img = img.numpy()
        
        return img, class_label

class ChessLoader:
    def __init__(self, dataset_path, use_augmentation, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_classes = 4  # Number of chess piece types to classify
        
        train_dataset = ChessPieceDataset(self.dataset_path, split='train', transform=get_transforms(105))
        val_dataset = ChessPieceDataset(self.dataset_path, split='val', transform=get_transforms(105))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_iterator = iter(self.train_loader)
    
    def get_train_batch(self):
        """Get a batch of training data"""
        try:
            # Try to get the next batch
            images, labels = next(self.train_iterator)
        except StopIteration:
            # If we've reached the end of the dataset, reset the iterator
            self.train_iterator = iter(self.train_loader)
            images, labels = next(self.train_iterator)
        
        # Convert labels to one-hot encoding
        one_hot_labels = to_categorical(labels, num_classes=self.num_classes)
        
        return images.numpy(), one_hot_labels
    
    def evaluate(self, model):
        """Evaluate the model on the validation set"""
        total_correct = 0
        total_samples = 0
        
        # Process validation set in batches
        for images, labels in self.val_loader:
            # Convert images to numpy and make predictions
            images_np = images.numpy()
            predictions = model.predict(images_np)
            
            # Get the predicted class indices
            pred_indices = np.argmax(predictions, axis=1)
            
            # Count correct predictions
            total_correct += np.sum(pred_indices == labels.numpy())
            total_samples += len(labels)
        
        # Return accuracy
        accuracy = total_correct / total_samples
        print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy
