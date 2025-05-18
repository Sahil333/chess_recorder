import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math

class SiameseNetwork(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=4096):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Initialize with specified random seeds
        # Following architecture from Koch et al. paper
        self.features = nn.Sequential(
            # Conv1: 64 filters of 10x10
            nn.Conv2d(input_channels, 64, kernel_size=10, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Conv2: 128 filters of 7x7
            nn.Conv2d(64, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Conv3: 128 filters of 4x4
            nn.Conv2d(128, 128, kernel_size=4, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Conv4: 256 filters of 4x4
            nn.Conv2d(128, 256, kernel_size=4, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # Determine feature dimensions after convolutions and flattening
        # For 105x105 input, after all conv and pooling layers: 256 * 2 * 2 = 1024
        self.fc = nn.Sequential(
            nn.Linear(9216, embedding_dim),
            nn.Sigmoid()
        )
        
        # Parameter for weighted L1 distance between embeddings
        self.alpha = nn.Parameter(torch.randn(embedding_dim) * 0.1)
        
        # Initialize weights as specified in the paper
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights according to the paper:
        - Conv weights: normal with mean=0, std=0.01
        - Conv biases: normal with mean=0.5, std=0.01
        - FC weights: normal with mean=0, std=0.01
        - FC biases: normal with mean=0.5, std=0.01
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.5, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.2)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.5, std=0.01)
    
    def forward_one(self, x):
        """Forward pass for one input"""
        x = self.features(x)
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        """Forward pass for a pair of inputs"""
        # Get embeddings for both inputs
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        
        # Calculate L1 weighted distance
        abs_diff = torch.abs(embedding1 - embedding2)
        weighted_diff = self.alpha * abs_diff
        
        # Sum with weights and apply sigmoid
        distance = torch.sum(weighted_diff, dim=1, keepdim=True)
        similarity = torch.sigmoid(distance)
        
        return similarity


class SiameseContrastiveLoss(nn.Module):
    """
    Contrastive loss function as described in the paper
    L(y,p) = y * log(p) + (1-y) * log(1-p) + λ*||w||²
    
    Using different regularization weights for different layer types:
    - Conv layers: lambda = 2e-4
    - FC layers: lambda = 2e-3
    """
    def __init__(self, lambda_reg=0.0001):
        super(SiameseContrastiveLoss, self).__init__()
        self.lambda_reg = lambda_reg
        # Using different regularization weights as in GitHub implementation
        self.conv_reg = 2e-4  # Regularization weight for Conv layers
        self.fc_reg = 2e-3    # Regularization weight for FC layers
    
    def forward(self, predictions, labels, model):
        # Ensure numerical stability
        eps = 1e-12
        predictions = torch.clamp(predictions, min=eps, max=1-eps)
        
        # Binary cross-entropy loss
        bce_loss = - torch.mean(
            labels * torch.log(predictions) + 
            (1 - labels) * torch.log(1 - predictions)
        )
        
        # L2 regularization term with different weights by layer type
        l2_reg = 0.0
        for name, param in model.named_parameters():
            # Skip biases and batch norm parameters for regularization
            if 'bias' in name or 'bn' in name:
                continue
                
            # Apply different regularization weights based on layer type
            if 'conv' in name:
                l2_reg += self.conv_reg * torch.sum(param ** 2)
            elif 'fc' in name or 'linear' in name:
                l2_reg += self.fc_reg * torch.sum(param ** 2)
            else:
                # For other parameters including the alpha weights
                l2_reg += self.lambda_reg * torch.sum(param ** 2)
        
        # Total loss
        total_loss = bce_loss + l2_reg
        
        return total_loss


# Transformation pipeline
def get_transforms(img_size=105):
    """
    Creates transformation pipeline as per the paper:
    - Resize to 105x105
    - Normalize
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Grayscale(num_output_channels=1)
    ])