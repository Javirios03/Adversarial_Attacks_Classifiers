import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

# Import all models
from models.AllConv import AllConv, AllConv_K5, AllConv_K7
from models.NiN import NiN
from models.VGG16 import VGG16

ORIGINAL_ACCS = {
    'allconv': 85.6,
    'allconv_k5': 85.6,
    'allconv_k7': 85.6,
    'nin': 87.2,
    'vgg16': 83.3
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['allconv', 'nin', 'vgg16', 'allconv_k5', 'allconv_k7'], required=True, help='Model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--resume_lr', type=float, default=0.01, help='Learning rate for resuming training')

    return parser.parse_args()


def get_model(model_name, num_classes=10):
    return {
        'allconv': AllConv(num_classes=num_classes),
        'allconv_k5': AllConv_K5(num_classes=num_classes),
        'allconv_k7': AllConv_K7(num_classes=num_classes),
        'nin': NiN(num_classes=num_classes),
        'vgg16': VGG16(num_classes=num_classes)
    }.get(model_name, None)


def get_optimizer(model, model_name, lr) -> optim.Optimizer:
    if model_name == 'allconv' or model_name == 'allconv_k5' or model_name == 'allconv_k7':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif model_name == 'nin':
        return optim.SGD(model.parameters(),
        lr=lr, momentum=0.9, weight_decay=5e-4)
    elif model_name == 'vgg16':
        return optim.SGD(model.parameters(),
        lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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


# def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, writer):
#     model.train()
#     train_loss = 0.0
#     correct = 0
#     total = 0

#     loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
#     for batch_idx, (data, target) in enumerate(loop):
#         data, target = data.to(device), target.to(device)
        
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item() * data.size(0)
#         preds = output.argmax(dim=1)
#         correct += (preds == target).sum().item()
#         total += target.size(0)

#         # Log Batch Metrics
#         if batch_idx % 100 == 0:
#             writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
#             writer.add_scalar('Accuracy/train_batch', 100. * correct / total, epoch * len(train_loader) + batch_idx)
        
#         loop.set_postfix({
#             'Loss': f"{loss.item():.4f}",
#             'Accuracy': f"{100. * correct / total:.2f}%",
#         })

#     epoch_loss = train_loss / total
#     epoch_acc = 100. * correct / total

#     # Log Epoch Metrics
#     writer.add_scalar('Loss/train', epoch_loss, epoch)
#     writer.add_scalar('Accuracy/train', epoch_acc, epoch)
#     writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

#     print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

#     return epoch_loss, epoch_acc


# def evaluate(model, device, test_loader, criterion, writer, epoch):
#     model.eval()
#     correct = 0
#     total = 0
#     test_loss = 0.0

#     with torch.no_grad():
#         for data, target in tqdm(test_loader, desc="Evaluating"):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)

#             test_loss += loss.item() * data.size(0)
#             preds = output.argmax(dim=1)
#             correct += (preds == target).sum().item()
#             total += target.size(0)

#     acc = 100. * correct / total
#     avg_loss = test_loss / total

#     # Log Validation Metrics
#     writer.add_scalar('Loss/val', avg_loss, epoch)
#     writer.add_scalar('Accuracy/val', acc, epoch)
#     print(f"Test Accuracy: {acc:.2f}%")
#     return acc


def train(model, device, train_loader, test_loader, optimizer, criterion, epochs, model_name, scheduler=None):
    original_acc_reached = False
    model.train()
    best_acc = 0.0

    # Checkpoints
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # checkpoint_dir = f"./checkpoints/{model_name}_{timestamp}"
    # os.makedirs(checkpoint_dir, exist_ok=True)
    original_acc = ORIGINAL_ACCS[model_name]

    try:
        for epoch in range(epochs):
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for data, target in train_loop:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loop.set_postfix(loss=loss.item())

            test_acc = test(model, device, test_loader)

            # Adjust learning rate if applicable
            if (model_name == 'allconv' or model_name == 'allconv_k5' or model_name == 'allconv_k7') and scheduler:
                scheduler.step(test_acc)
            elif model_name == 'vgg16' and scheduler:
                scheduler.step()
            elif model_name == 'nin' and scheduler:
                scheduler.step()

            # Log test accuracy
            print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.5f}")

            # Save checkpoint once original accuracy is reached
            # if (epoch + 1) % 10 == 0 or test_acc > best_acc:
            if not original_acc_reached and test_acc >= original_acc:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler else None,
                    'accuracy': test_acc,
                    'loss': loss.item()
                }

                torch.save(checkpoint, f"./results/{model_name}_original_acc.pth")
                print(f"Original accuracy reached: {model_name} - {test_acc:.2f}%")
                original_acc_reached = True
                # Keep best model
                # if test_acc > best_acc:
                #     best_acc = test_acc
                #     torch.save(checkpoint, f"{checkpoint_dir}/best_model.pth")
                #     print(f"New best model saved with accuracy: {best_acc:.2f}%")

                # Periodic checkpoint
                # torch.save(checkpoint, f"{checkpoint_dir}/epoch_{epoch + 1}.pth")
                # print(f"Checkpoint saved for epoch {epoch + 1}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'accuracy': test_acc,
            'loss': loss.item()
        }
        torch.save(checkpoint, f"./results/{model_name}_interrupted.pth")
        print(f"Checkpoint saved for interrupted training at epoch {epoch + 1}")
    finally:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'accuracy': test_acc,
            'loss': loss.item()
        }
        torch.save(checkpoint, f"./results/{model_name}_final.pth")
        print(f"Final checkpoint saved for epoch {epoch + 1}")


def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)


def main():
    args = parse_args()

    # Load Data
    train_set, test_set = load_datasets()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)
    print(f"Model: {args.model}, Device: {device}")
    optimizer = get_optimizer(model, args.model, args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.model == 'allconv' or args.model == 'allconv_k5' or args.model == 'allconv_k7':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.2,
            patience=8,
            min_lr=1e-4
        )
    elif args.model == 'vgg16':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[25, 50, 75],
            gamma=0.1
        )
    elif args.model == 'nin':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[15, 30, 45],
            gamma=0.1
        )

    # Load checkpoint if resuming
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            if scheduler and 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}, accuracy: {checkpoint['accuracy']:.2f}%")
        else:
            print(f"No checkpoint found at '{args.resume}'")
            return

    # Train
    print(f"Training {args.model.upper()} model...")
    if args.model == 'allconv' or args.model == 'allconv_k5' or args.model == 'allconv_k7':
        print("Using ReduceLROnPlateau scheduler")
    elif args.model == 'vgg16':
        print("Using MultiStepLR scheduler: [25, 50, 75]")
    elif args.model == 'nin':
        print("Using MultiStepLR scheduler: [15, 30, 45]")

        # # Reset LR Scheduler
        # if scheduler:
        #     scheduler = optim.lr_scheduler.MultiStepLR(
        #         optimizer,
        #         milestones=[50, 75],
        #         gamma=0.1
        #     )
    if args.resume:
        train(model, device, train_loader, test_loader, optimizer, criterion, args.epochs - start_epoch, args.model, scheduler)
    else:
        train(model, device, train_loader, test_loader, optimizer, criterion, args.epochs, args.model, scheduler)
    print("Training complete!")

    # # Save the model
    # torch.save(model.state_dict(), f"./models/{args.model}.pth")


if __name__ == "__main__":
    main()

    # Example to resume: python -m scripts.train --model vgg16 --resume "checkpoints/vgg16_20250504-114620/epoch_50.pth"
