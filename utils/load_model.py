from models.AllConv import AllConv
from models.NiN import NiN
from models.VGG16 import VGG16
from torch import nn
import torch
import argparse

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# checkpoints_dir = 'checkpoints/Best Models Yet'
# models_dir = 'models'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['allconv', 'nin', 'vgg16'], required=True, help='Model to obtain')
    parser.add_argument('--chckpt_path', type=str, help='Path to the checkpoint file', required=True)
    return parser.parse_args()


def get_model(model_name, num_classes=10):
    if model_name == 'allconv':
        return AllConv(num_classes=num_classes)
    elif model_name == 'nin':
        return NiN(num_classes=num_classes)
    elif model_name == 'vgg16':
        return VGG16(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_accuracy(model):
    """
    Auxiliary function to test a given model on the CIFAR10. The dataset
    must be already downloaded and correctly stored in the data folder.
    """
    # Load Dataset - Specific to CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Evaluate the model
    correct, total = 0, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100. * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')


if __name__ == '__main__':
    args = parse_args()
    model = get_model(args.model)

    # Load the model
    checkpoint = torch.load(args.chckpt_path, map_location='cpu')

    try:
        model.load_state_dict(checkpoint['model_state'])
    except KeyError:
        # The checkpoint wasn't saved as a dictionary with model state a key
        model.load_state_dict(checkpoint)

    # Get the model accuracy
    get_accuracy(model)

    # How to use this script: 
    # python -m utils.load_model --model allconv --chckpt_path ./models/allconv.pth

    # Path must be relative to the root of the project (the one containing scripts, models, utils... folders)
