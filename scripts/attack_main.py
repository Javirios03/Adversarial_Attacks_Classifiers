import torch
# import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image
import os
# import time

# Auxiliary scripts
from scripts.OPA_funcs import OnePixelAttack
from utils.attack_aux_funcs import visualize_perturbations, CIFAR_LABELS
from config import IMAGES, MODELS_DICT, PRETRAINED_MODELS

# IMAGES_PATH = './data/images'


def save_original_image(img, label, model_name, idx):
    path = os.path.join(IMAGES, model_name, CIFAR_LABELS[label], f'{idx}_original.png')
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(img.cpu(), path)


def load_used_indices(model_name) -> set:
    """
    Auxiliary function to obtain those indices, corresponding to the images in the test dataset, for which an adversarial attack has already been performed.

    Parameters
        - model_name (str): Name of the model used for the attack. In our case, it can be 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16' or  'original_vgg16'

    Returns
        - used_indices (set): Set of indices corresponding to the images in the test dataset for which an adversarial attack has already been performed.
    """
    if os.path.exists(f'{IMAGES}/{model_name}/used_indices.txt'):
        with open(f'{IMAGES}/{model_name}/used_indices.txt', 'r') as f:
            return set(map(int, f.read().splitlines()))
    else:
        return set()


def save_used_indices(model_name, used_indices):
    """
    Auxiliary function to save the indices of the images in the test dataset for which an adversarial attack has already been performed. Updates (or creates, if not existant) the file corresponding to the record.

    Parameters
        - model_name (str): Name of the model used for the attack. It can be 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16' or  'original_vgg16'
        - used_indices (set): Set of indices corresponding to the images in the test dataset for which an adversarial attack has already been performed.
    """
    os.makedirs(IMAGES, exist_ok=True)
    with open(f'{IMAGES}/{model_name}/used_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, sorted(used_indices))))


def load_model(model_name, model_path, device):
    """
    Loads the model specified by model_name and moves it to the specified device.

    Parameters
        - model_name (str): Name of the model to load. It can be 'AllConv', 'NiN' or 'VGG16'.
        - model_path (str): Path to the pre-trained model weights.
        - device (str): Device to load the model on ('cuda' or 'cpu').

    Returns
        - model (torch.nn.Module): The loaded model.
    """
    try:
        model = MODELS_DICT[model_name]()
    except KeyError:
        raise ValueError(f"Model {model_name} is not recognized. Available models: {list(MODELS_DICT.keys())}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def get_valid_images(dataset, model_name, device, num_images=10):
    """
    Samples *num_images* images from the provided dataset, such that the model's prediction is correct and the image has not been used in a previous attack.
    
    Parameters
        - dataset (torch.utils.data.Dataset): Dataset from which to sample images.
        - model_name (str): Name of the model used for the attack. It can be 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16' or  'original_vgg16'.
        - device (str): Device to use for computation ('cuda' or 'cpu').
        - num_images (int): Number of images to sample."""
    model = load_model(model_name, PRETRAINED_MODELS[model_name], device)
    used_indices = load_used_indices(model_name)
    valid_images, attempts_count = [], 0
    max_attempts = 5*num_images  # Limit attempts to avoid infinite loop

    while len(valid_images) < num_images and attempts_count < max_attempts:
        idx = torch.randint(0, len(dataset), (1,)).item()
        if idx in used_indices:
            attempts_count += 1
            continue

        img, label = dataset[idx]
        img = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(img.unsqueeze(0)), dim=1).item()
        if pred == label:  # Correct prediction
            valid_images.append((img, label, idx))
            used_indices.add(idx)
        attempts_count += 1

    if len(valid_images) < num_images:
        print(f"Warning: Only {len(valid_images)} valid images found out of {num_images} requested. Consider increasing the dataset size or reducing the number of images requested.")

    save_used_indices(model_name, used_indices)
    return valid_images


def main(model_name, n=400, epochs=100, num_images=10, device='cuda'):
    model_path = PRETRAINED_MODELS.get(model_name, None)
    if model_path is None:
        raise ValueError(f"Model {model_name} is not recognized. Available models: {list(PRETRAINED_MODELS.keys())}")
    # Load the model
    model = load_model(model_name, model_path, device)
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model: {model_name} from {model_path}")
    print(f"Model architecture: {model}")

    # Load CIFAR-10 dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=test_transform)

    # # Obtain a random image and its label
    # idx = int(torch.randint(0, len(test_dataset), (1,)).item())
    # img, label = test_dataset[idx]  # label is an integer from 0 to 9

    # # Convert to tensor and move to device
    # img = img.to(device)

    # with torch.no_grad():
    #     original_pred = torch.argmax(model(img.unsqueeze(0).to(device)), dim=1).item()
    # # print(model(img.unsqueeze(0).to(device)))  # Raw logits for each class. - Example: tensor([[-6.2348, -5.0969, -2.6802, 20.7050, -6.4521,  9.5229,  1.6183, -2.0581, -7.1109, -8.2980]], grad_fn=<ViewBackward0>)
    # # print(F.softmax(model(img.unsqueeze(0).to(device)), dim=1))  # Related probabilities for each class. - Example for previous logits: tensor([[1.9961e-12, 6.2285e-12, 6.9814e-11, 9.9999e-01, 1.6063e-12, 1.3922e-05, 5.1375e-09, 1.3005e-10, 8.3125e-13, 2.5361e-13]], grad_fn=<SoftmaxBackward0>)
    # # print(torch.argmax(model(img.unsqueeze(0).to(device)), dim=1))  # Original prediction. - Example: tensor([3])
    # print(f"Original prediction: {original_pred}")  # Predicted class: 3
    # print(f"Original label: {label}")  # Real label: 3

    # for target_label in range(10):
    #     if target_label != label:
    #         print(f"Trying to change label {label} to {target_label} with classes {CIFAR_LABELS[label]} to {CIFAR_LABELS[target_label]}")
    #         # Run attack
    #         start = time.perf_counter()
    #         attack = OnePixelAttack(model, img, label, target_label, n=n)
    #         perturbed_img, _ = attack.perturb_img(epochs=epochs, d=1, show=True, print_every=10, title=f"One Pixel Attack on {model_name} - {n} pixels")

    #         attack_time = time.perf_counter()
    #         print(f"Attack time: {attack_time - start:.2f} seconds")

    #         # Verify attack success
    #         with torch.no_grad():
    #             perturbed_pred = torch.argmax(model(perturbed_img.unsqueeze(0).to(device)), dim=1).item()

    #         verify_time = time.perf_counter()
    #         print(f"Verification time: {verify_time - attack_time:.2f} seconds")

    #         print(f"Original prediction: {original_pred}, Perturbed prediction: {perturbed_pred}")

    #         # Visualize perturbations
    #         visualize_perturbations(perturbed_img, img, label, model, title=f"OPA - {model_name} - Original {label}-{original_pred} - Tried {target_label}-{perturbed_pred}")

    #         visualize_time = time.perf_counter()
    #         print(f"Visualization time: {visualize_time - verify_time:.2f} seconds")
    #         print(f"Total time: {visualize_time - start:.2f} seconds")

    # Select new images for the attack
    samples = get_valid_images(test_dataset, model_name, device, num_images=num_images)
    print(f"Selected {len(samples)} valid images for the attack.")

    for img, label, idx in samples:
        print(f"\n--- Attacking image {idx} (True Label: {label}) ---")

        # Save the original image
        save_original_image(img, label, model_name, idx)

        original_pred = torch.argmax(model(img.unsqueeze(0).to(device)), dim=1).item()

        for target_label in range(10):
            if target_label != label:
                print(f"Targeting class {target_label} ({CIFAR_LABELS[target_label]}) to change from class {label} ({CIFAR_LABELS[label]})")

                # Run attack
                # start = time.perf_counter()
                attack = OnePixelAttack(model, img, label, target_label, n=n)
                perturbed_img, _ = attack.perturb_img(epochs=epochs, d=1, show=False, print_every=20)

                # attack_time = time.perf_counter()
                # print(f"Attack time: {attack_time - start:.2f} seconds")

                # Verify attack success
                with torch.no_grad():
                    perturbed_pred = torch.argmax(model(perturbed_img.unsqueeze(0).to(device)), dim=1).item()

                # verify_time = time.perf_counter()
                # print(f"Verification time: {verify_time - attack_time:.2f} seconds")

                print(f"Original prediction: {original_pred}, Perturbed prediction: {perturbed_pred}")

                # Visualize perturbations
                visualize_perturbations(
                    perturbed_img, img, label, model,
                    model_name=model_name,
                    idx=idx,
                    target_label=target_label,
                    perturbed_label=perturbed_pred
                )

                # visualize_time = time.perf_counter()
                # print(f"Visualization time: {visualize_time - verify_time:.2f} seconds")
                # print(f"Total time: {visualize_time - start:.2f} seconds")


if __name__ == "__main__":
    # Example usage
    model_name = 'conv_vgg16'  # Choose from 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16', 'original_vgg16'
    # model_path = f'./results/allconv.pth'
    n = 400  # Initial population size for DE
    # batch_size = 100
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Using device: {device}")
    main(model_name, n=n, epochs=epochs, num_images=1, device=device)
