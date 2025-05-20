import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.AllConv import AllConv
from tqdm import tqdm
import os

# Hyperparameters
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100


def load_datasets():
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    print("Loading CIFAR-10 dataset...")
    # Check if data exists before loading
    data_root = './data/cifar-10-batches-py'
    if not os.path.exists(data_root):
        print("Downloading CIFAR-10...")
        datasets.CIFAR10(root='./data', train=True, download=True)  # Download only once
        datasets.CIFAR10(root='./data', train=False, download=True)

    # Load datasets without download verification messages
    train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

    return train_set, test_set


# Training function
def train(model, device, train_loader, test_loader, optimizer, criterion):
    print(f"Training on {device}...")
    model.train()
    for epoch in range(EPOCHS):
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for data, target in train_loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())
        
        # Validation
        test_acc = test(model, device, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Test Accuracy: {test_acc:.2f}%")


# Testing function
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc


def main():
    # Load datasets
    train_set, test_set = load_datasets()

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize other variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AllConv(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate the model
    train(model, device, train_loader, test_loader, optimizer, criterion)

    # Final test accuracy
    final_test_acc = test(model, device, test_loader)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "./models/allconv.pth")
    print("Model saved to ./models/allconv.pth")


if __name__ == "__main__":
    main()