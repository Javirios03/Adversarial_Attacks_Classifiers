import PIL.Image
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import pandas as pd
import os
import random
import io
import contextlib
import warnings
from collections import defaultdict
# from torchvision.utils import save_image
from config import CIFAR_LABELS, MODELS_DICT, PRETRAINED_MODELS, IMAGES, CIFAR_10_MEAN, CIFAR_10_STD


def load_used_indices(model_name) -> set:
    """
    Auxiliary function to obtain those indices, corresponding to the images in the test dataset, for which an adversarial attack has already been performed, as well as their corresponding labels. This is used to avoid repeating attacks on the same images.

    Parameters
        - model_name (str): Name of the model used for the attack. In our case, it can be 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16' or  'original_vgg16'

    Returns
        - used_indices (set(tuple)): Set of tuples (index, label) corresponding to the images in the test dataset for which an adversarial attack has already been performed.
            * index (int): Index of the image in the test dataset.
            * label (int): Label of the image in the test dataset (0-9, which can be converted to its class name using CIFAR_LABELS).
    """
    path = f'{IMAGES}/{model_name}/used_indices.txt'
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as f:
        return set(tuple(map(int, line.strip().split(','))) for line in f.readlines())


def save_used_indices(model_name, used_indices):
    """
    Auxiliary function to save the indices of the images in the test dataset for which an adversarial attack has already been performed. Updates (or creates, if not existant) the file corresponding to the record.

    Parameters
        - model_name (str): Name of the model used for the attack. It can be 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16' or  'original_vgg16'
        - used_indices (set(tuple)): Set of tuples (index, label) corresponding to the images in the test dataset for which an adversarial attack has already been performed. Both index and label are integers.
    """
    os.makedirs(IMAGES, exist_ok=True)
    path = f'{IMAGES}/{model_name}/used_indices.txt'
    with open(path, 'w') as f:  # Creates or overwrites the file. Order is guaranteed, since any used_indices contains, by default, the already present indices.
        f.writelines(f"{idx},{label}\n" for idx, label in sorted(used_indices))


def get_valid_images(dataset: torchvision.datasets.CIFAR10, model_name: str, device: str = 'cpu', num_images=10, force_index=None) -> List[Tuple[torch.Tensor, int, int]]:
    """
    Samples *num_images* images from the provided dataset, such that the model's prediction is correct and the image has not been used in a previous attack.

    Parameters
        - dataset (torchvision.datasets.CIFAR10): The dataset from which to sample images. Hard-coded for CIFAR-10.
        - model_name (str): Name of the model used for the attack. It can be 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16' or  'original_vgg16'.
        - device (str): Device to use for computation ('cuda' or 'cpu').
        - num_images (int): Number of images to sample.

    Returns
        - valid_images (list): List of tuples (img, label, idx) where:
            * img (torch.Tensor): The image tensor, no normalization applied. In range [0, 1].
            * label (int): The label of the image (0-9, which can be converted to its class name using CIFAR_LABELS).
            * idx (int): The index of the image in the dataset.
    """
    if num_images <= 0:
        warnings.warn("num_images should be greater than 0. Returning an empty list.", UserWarning)
        return []
    elif num_images < 10:
        warnings.warn("num_images is less than 10. Since class diversity is imposed, this may lead to fewer images being selected than requested.", UserWarning)
    model = load_model(model_name, device)
    used_indices = load_used_indices(model_name)
    valid_images, attempts_count = [], 0
    class_counts = defaultdict(int)  # To track how many images per class have been selected
    max_attempts = 10*num_images  # Limit attempts to avoid infinite loop
    per_class_limit = num_images // 10  # Limit per class to ensure diversity (hard-coded for CIFAR-10)
    if per_class_limit == 0:
        per_class_limit = 1  # Ensure at least one image per class if num_images < 10

    if force_index is not None:
        return [(dataset[force_index][0].to(device), dataset[force_index][1], force_index)]
    while len(valid_images) < num_images and attempts_count < max_attempts:
        # Only use the second half of the dataset (Colab used the first half)
        idx = torch.randint(len(dataset)//2, len(dataset), (1,)).item()
        img, label = dataset[idx]
        if (idx, label) in used_indices or class_counts[label] >= per_class_limit:
            attempts_count += 1
            continue

        img: torch.Tensor = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(normalize_cifar10(img).unsqueeze(0).to(device)), dim=1).item()
        if pred == label:  # Correct prediction
            valid_images.append((img, label, idx))
            used_indices.add((idx, label))
            class_counts[label] += 1
        attempts_count += 1

    if len(valid_images) < num_images:
        print(f"Warning: Only {len(valid_images)} valid images found out of {num_images} requested. Consider increasing the dataset size or reducing the number of images requested.")

    save_used_indices(model_name, used_indices)
    return valid_images


def load_model(model_name, device):
    """
    Loads the model specified by model_name and moves it to the specified device.

    Parameters
        - model_name (str): Name of the model to load. It can be 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16', or 'original_vgg16'.
        - device (str): Device to load the model on ('cuda' or 'cpu').

    Returns
        - model (torch.nn.Module): The loaded model.
    """
    try:
        model = MODELS_DICT[model_name]()
    except KeyError:
        raise ValueError(f"Model {model_name} is not recognized. Available models: {list(MODELS_DICT.keys())}")
    model_path = PRETRAINED_MODELS.get(model_name)
    if model_path is None:
        raise ValueError(f"No pre-trained model path found for {model_name}. Please check the configuration.")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except KeyError:
        # The checkpoint wasn't saved as a dictionary with model state a key
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model state from checkpoint: {e}")
    model.to(device)
    return model


def normalize_cifar10(img_tensor: torch.Tensor) -> torch.Tensor:
    mean, std = CIFAR_10_MEAN.to(img_tensor.device), CIFAR_10_STD.to(img_tensor.device)
    return (img_tensor - mean) / std


def _add_information(
    img: torch.Tensor, perturbed_img: torch.Tensor, label: int, model: nn.Module
) -> str:
    """
    Adds information to the image corresponding to the "before and after" of the
    perturbation.

    Parameters
    ----------
    img           : Original image. Dimensions: [channels, height, width].
    perturbed_img : Perturbed image. Dimensions: [channels, height, width].
    label         : Real label of the image.
    model         : Model used.

    Returns
    -------
    String with the included information.
    """

    real_class = CIFAR_LABELS[label]
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        print(f"True Label: {real_class}")
        with torch.no_grad():
            y_pred = F.softmax(model(normalize_cifar10(img.unsqueeze(0)).squeeze()), dim=0)
            y_pred_perturbed = F.softmax(
                model(normalize_cifar10(perturbed_img.unsqueeze(0)).squeeze()), dim=0
            )

        original_prediction = CIFAR_LABELS[torch.argmax(y_pred)]
        original_prob_label = y_pred[label].item()
        original_prob = torch.max(y_pred).item()
        print(f"\nOriginal prediction: {original_prediction}")
        print(
            f"Probability for original label ({real_class}):"
            f"{original_prob_label:.2f}"
        )
        print(
            f"Probability for original prediction ({original_prediction}):"
            f"{original_prob:.2f}"
        )

        perturbed_prediction = CIFAR_LABELS[torch.argmax(y_pred_perturbed)]
        perturbed_prob_label = y_pred_perturbed[label].item()
        perturbed_prob = torch.max(y_pred_perturbed).item()
        print(f"\nPerturbed prediction: {perturbed_prediction}")
        print(
            f"Probability for original prediction ({original_prediction}):"
            f"{perturbed_prob_label:.2f}"
        )
        print(
            f"Probability for perturbed prediction ({perturbed_prediction}):"
            f"{perturbed_prob:.2f}"
        )

    return output.getvalue()


def check_logs(csv_log_path: str) -> None:
    """
    Checks the logs in the provided CSV file and provides a statistical summary

    Parameters
        - csv_log_path (str): Path to the CSV file containing the logs.

    Returns
        - None: Prints the summary statistics to the console.
    """
    df = pd.read_csv(csv_log_path)

    # Obtain total astats
    total_images = len(df)
    total_successful_attacks = df['success'].sum()
    suc_rate = 100 * total_successful_attacks / total_images if total_images > 0 else 0

    print(f"Total images: {total_images}")
    print(f"Total successful attacks: {total_successful_attacks}")
    print(f"Success rate: {suc_rate:.2f}%")

    # Obtain stats per class
    print("\nSuccess rate per original label:")
    print(df.groupby('orig_label')['success'].mean().mul(100).round(2).sort_values(ascending=False))

    print("\nSuccess rate by Target Label:")
    print(df.groupby('target_label')['success'].mean().mul(100).round(2).sort_values(ascending=False))

    # Non-targeted attacks
    grouped_success = df.groupby('image_idx')['success'].any()
    non_targeted_success_rate = grouped_success.mean() * 100
    print(f"\nNon-targeted success rate (at least one success per image): {non_targeted_success_rate:.2f}%")


def visualize_perturbations(
    perturbed_img: torch.Tensor,
    img: torch.Tensor,
    label: int,
    model: nn.Module,
    model_name: str,
    idx: int,
    target_label=None,
    perturbed_label=None
) -> None:
    """
    Saves a figure with the "before and after" of the perturbation.

    Parameters
    ----------
        - perturbed_img (torch.Tensor): Image with the pixel/s changed (attack result).
        - img (torch.Tensor): Original image.
        - label (int): Real label of the image (0-9)
        - model (nn.Module): Model used.
        - model_name (str): Name of the model used.
        - idx (int): Index of the image in the dataset.
        - target_label (int): Target label for the attack, if applicable.
        - perturbed_label (int): Actual label achieved (determined by whether the attack was successful or not).
    """

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(
        _add_information(img, perturbed_img, label, model),
        ha="center",
        fontsize=14,
        fontfamily="monospace",
        wrap=True,
        y=1.05,
    )

    axs[0].imshow(
        np.transpose(img.cpu().numpy(), (1, 2, 0)), interpolation="nearest"
    )  # type: ignore
    axs[0].set_title("Original Image")  # type: ignore
    axs[1].imshow(
        np.transpose(perturbed_img.cpu().numpy(), (1, 2, 0)),
        interpolation="nearest",
    )  # type: ignore
    axs[1].set_title("Perturbed Image")  # type: ignore
    plt.subplots_adjust(top=0.8)

    # Save under the correct directory
    class_dir = CIFAR_LABELS[label]
    out_dir = os.path.join(
        "data", "images", model_name, class_dir)
    os.makedirs(out_dir, exist_ok=True)

    if target_label is not None:
        filename = f"img_{idx}_target_{target_label}_achieved_{perturbed_label}.png"
    else:
        filename = f"img_{idx}_label_{label}.png"

    # Save the pair of images with the information
    plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)

    # Save, also, the perturbed image by itself
    # perturbed_out_dir = os.path.join(
    #     "data", "images", model_name, class_dir)
    # # os.makedirs(perturbed_out_dir, exist_ok=True)
    # filename = f"{idx}_target_{target_label}_achieved_{perturbed_label}.png" if target_label is not None else f"{idx}.png"
    # save_image(perturbed_img, os.path.join(perturbed_out_dir, filename))


def save_img_cutmix(
    images: torch.Tensor,
    labels: tuple[int, int],
    cutmix_img: torch.Tensor,
    cutmix_label: float,
    title=None,
) -> None:
    """
    Saves a figure for the "before and after" of the CutMix.

    Parameters
    ----------
    images       : Original images. Dimensions: [2, channels, height, width].
    labels       : Original labels.
    cutmix_img   : Modified image. Dimensions: [channels, height, width].
    cutmix_label : Modified label.
    """

    _, axs = plt.subplots(1, 3, figsize=(14, 8))

    axs[0].imshow(
        np.transpose(images[0].cpu().numpy(), (1, 2, 0)), interpolation="nearest"
    )  # type: ignore
    axs[0].set_title(f"Original Image. Label: {labels[0]}.")  # type: ignore

    axs[1].imshow(
        np.transpose(images[1].cpu().numpy(), (1, 2, 0)),
        interpolation="nearest",
    )  # type: ignore
    axs[1].set_title(f"Original Image. Label: {labels[1]}.")  # type: ignore

    axs[2].imshow(
        np.transpose(cutmix_img.cpu().numpy(), (1, 2, 0)),
        interpolation="nearest",
    )  # type: ignore
    axs[2].set_title(f"CutMix Image. Label: {cutmix_label:.2f}.")  # type: ignore

    if title is None:
        title = "CutMix"
    plt.savefig(f"images/{title}.png", bbox_inches="tight")


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Parameters
    ----------
    seed : Seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
