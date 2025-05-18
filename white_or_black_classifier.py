import os
import glob
import argparse
import random

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# New Dataset class using the prepared dataset structure
class ChessPieceDatasetPrepared(Dataset):
    def __init__(self, root_dir, split, transform=None):
        """
        root_dir: Root directory of the prepared dataset.
        split: One of 'train', 'valid', or 'test'.
        transform: torchvision transforms to apply.
        """
        self.transform = transform
        images_dir = os.path.join(root_dir, split, "images")
        labels_dir = os.path.join(root_dir, split, "labels")
        self.labels_dir = labels_dir
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        base = os.path.basename(image_path)
        # Expecting a corresponding label file with the same basename and a .txt extension.
        label_path = os.path.join(self.labels_dir, base.replace(".jpg", ".txt"))
        try:
            with open(label_path, "r") as f:
                line = f.readline().strip()
            parts = line.split()
            label = int(parts[0])
        except Exception as e:
            raise RuntimeError(f"Error reading label file for {image_path}: {e}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# A small ConvNet inspired by handwritten digit classifiers.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input images are 3 channels (RGB)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # After two pool layers, 75x75 becomes approximately 18x18 (if odd, floor division is applied)
        self.fc1 = nn.Linear(64 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: white and black

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (32, ~37, ~37)
        x = self.pool(F.relu(self.conv2(x)))   # -> (64, ~18, ~18)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}] finished, Average Loss: {avg_loss:.4f}")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            # Get the index of the max log-probability (for classification)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return test_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Chess Piece White/Black Classifier Training")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the prepared dataset directory containing 'train' and 'valid' splits with 'images' and 'labels' folders")
    parser.add_argument("--save_model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()

    # Define transforms: resize to 75x75, convert to tensor, and normalize.
    transform = transforms.Compose([
        transforms.Resize((75, 75)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets using the prepared dataset structure.
    train_dataset = ChessPieceDatasetPrepared(root_dir=args.dataset_dir, split="train", transform=transform)
    valid_dataset = ChessPieceDatasetPrepared(root_dir=args.dataset_dir, split="valid", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        _, accuracy = test(model, device, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            if args.save_model:
                torch.save(model.state_dict(), "best_white_black_classifier.pth")
                print("Saved Best Model with accuracy: {:.2f}%".format(best_acc))

if __name__ == "__main__":
    main()