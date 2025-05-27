import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import io
import contextlib
from torchvision.utils import save_image


CIFAR_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


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
            y_pred = F.softmax(model(img.unsqueeze(0)).squeeze(), dim=0)
            y_pred_perturbed = F.softmax(
                model(perturbed_img.unsqueeze(0)).squeeze(), dim=0
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
    perturbed_out_dir = os.path.join(
        "data", "images", model_name, class_dir)
    # os.makedirs(perturbed_out_dir, exist_ok=True)
    filename = f"{idx}_target_{target_label}.png" if target_label is not None else f"{idx}.png"
    save_image(perturbed_img, os.path.join(perturbed_out_dir, filename))


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
