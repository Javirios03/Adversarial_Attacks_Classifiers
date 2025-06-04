# from models.AllConv import AllConv
# from models.NiN import NiN
# from models.VGG16 import VGG16
from torch import nn
import torch
import argparse
from config import IMAGES, MODELS_DICT, TEST_SET, TEST_TRANSFORM, CIFAR_LABELS
from utils.attack_aux_funcs import normalize_cifar10
import os

from torch.utils.data import DataLoader
from PIL import Image

# checkpoints_dir = 'checkpoints/Best Models Yet'
# models_dir = 'models'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['nin', 'conv_allconv', 'original_allconv', 'conv_vgg16', 'original_vgg16'],
                        help='Name of the model to load', required=True)
    parser.add_argument('--chckpt_path', type=str, help='Path to the checkpoint file', required=True)
    return parser.parse_args()


def load_perturbation(file_path: str, img: torch.Tensor) -> torch.Tensor:
    """
    Loads the perturbation from a .pt file. The file must contain the keys 'row', 'col', and 'rgb',
    which represent the row and column indices of the pixel to be changed, and the RGB values to set.

    Parameters
        - file_path (str): Path to the .pt file containing the perturbation.
        - img (torch.Tensor): The original image to which the perturbation will be applied.

    Returns
        - perturbed_img (torch.Tensor): The original image with the perturbation applied.        
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    p_img = torch.clone(img)

    perturbation = torch.load(file_path, map_location='cpu')
    row = perturbation['row']
    col = perturbation['col']
    p_img[:, row, col] = perturbation['rgb']
    return p_img


def undo_normalization_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def get_model(model_name, checkpoint_path, device: str, num_classes=10):
    try:
        model = MODELS_DICT[model_name](num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Model '{model_name}' is not recognized. Available models: {list(MODELS_DICT.keys())}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    try:
        model.load_state_dict(checkpoint['model_state'])
    except KeyError:
        # The checkpoint wasn't saved as a dictionary with model state a key
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model state from checkpoint: {e}")

    return model


def get_accuracy(model):
    """
    Auxiliary function to test a given model on the CIFAR10. The dataset
    must be already downloaded and correctly stored in the data folder.
    """
    # Create a DataLoader for the test set
    test_loader = DataLoader(TEST_SET, batch_size=128, shuffle=False)

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


def check_adversarial_samples(model, model_name, label: str, device='cpu'):
    model_images_directory = os.path.join(IMAGES, model_name, label)
    if not os.path.exists(model_images_directory):
        raise FileNotFoundError(f"Directory {model_images_directory} does not exist.")

    # List the different images (idx) in the directory if starts with number, not a character
    images = [f for f in os.listdir(model_images_directory) if f[0].isdigit() and f.endswith('.png')]
    if not images:
        raise FileNotFoundError(f"No images found in {model_images_directory}.")
    images.sort()  # Sort to ensure consistent order
    print(f"Found {len(images)} images in {model_images_directory}.")
    print(f"Images: {images}")

    # Pass the original image ({idx}_original.png) through the model
    for image_name in images:
        idx = image_name.split('_')[0]
        if 'original' in image_name:

            # Obtain the image in the test set
            original_image = TEST_SET[int(idx)][0].unsqueeze(0).to(device)
            original_label = TEST_SET[int(idx)][1]
            print(f"Original image {idx} label: {original_label}")

            # Get the model prediction
            model.eval()
            with torch.no_grad():
                output = model(original_image)
                pred_label = output.argmax(dim=1).item()
                print(f"Model prediction for original image {idx}: {pred_label}, original label: {original_label}")
            
            # Obtain the original image as saved in the directory
            original_image_path = os.path.join(model_images_directory, image_name)
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Image {original_image_path} does not exist.")
            original_image_saved = TEST_TRANSFORM(Image.open(original_image_path).convert('RGB')).unsqueeze(0).to(device)
            original_img_unnorm = undo_normalization_cifar10(original_image_saved).squeeze(0)
            with torch.no_grad():
                output_saved = model(original_image_saved)
                pred_label_saved = output_saved.argmax(dim=1).item()
                print(f"Model prediction for saved original image {idx}: {pred_label_saved}, original label: {original_label}")
        else:
            values = image_name.split('_')
            # Obtain the perturbed image ({idx}_perturbed.png) in the test set
            perturbed_image = TEST_TRANSFORM(Image.open(os.path.join(model_images_directory, image_name)).convert('RGB')).unsqueeze(0).to(device)
            perturbed_image_unnorm = undo_normalization_cifar10(perturbed_image).squeeze(0)
            perturbed_label = int(values[4].split('.')[0])
            print(f"Perturbed image {idx} label: {perturbed_label}")
            # Get the model prediction
            model.eval()
            with torch.no_grad():
                output = model(perturbed_image)
                pred_label = output.argmax(dim=1).item()
                print(f"Model prediction for perturbed image {idx}: {pred_label}, perturbed label: {perturbed_label}")

            # Calculate the difference between the original and perturbed images
            pixel_diff = (original_img_unnorm - perturbed_image_unnorm).abs()

            # Masking
            threshold = 1e-4
            mask = (pixel_diff > threshold).any(dim=0)  # Any channel differs
            num_changed_pixels = mask.sum().item()
            print(f"Number of changed pixels in image {idx}: {num_changed_pixels}")

            if num_changed_pixels > 1:
                print(f"Label {perturbed_label} has more than one pixel changed")
                changed = torch.nonzero(mask)
                for x, y in changed:
                    orig_pixel = original_img_unnorm[:, x, y]
                    perturbed_pixel = perturbed_image_unnorm[:, x, y]
                    print(f"Pixel changed at ({x.item()}, {y.item()}): "
                          f"Original: {[round(v.item(), 3) for v in orig_pixel]}, "
                          f"Perturbed: {[round(v.item(), 3) for v in perturbed_pixel]}")


if __name__ == '__main__':
    args = parse_args()
    model = get_model(args.model, args.chckpt_path, 'cpu')

    # Get the model accuracy
    # get_accuracy(model)

    # Check adversarial samples
    check_adversarial_samples(model, args.model, 'frog', 'cpu')

    # How to use this script: 
    # python -m utils.load_model --model conv_allconv --chckpt_path ./results/allconv.pth

    # Path must be relative to the root of the project (the one containing scripts, models, utils... folders)
