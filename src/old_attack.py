import torch
# import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image
import os
# import time
from collections import defaultdict
import warnings

# Auxiliary scripts
from scripts.OPA_funcs import OnePixelAttack
from utils.attack_aux_funcs import visualize_perturbations, normalize_cifar10, load_model, get_valid_images
from config import IMAGES, MODELS_DICT, PRETRAINED_MODELS, TEST_SET, CIFAR_LABELS

# IMAGES_PATH = './data/images'


def save_original_image(img, label, model_name, idx):
    path = os.path.join(IMAGES, model_name, CIFAR_LABELS[label], f'{idx}_original.png')
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(img.cpu(), path)


def save_perturbation(perturbations, model_name, idx, original_label, target_label, perturbed_pred):
    perturbed_path = os.path.join(IMAGES, model_name, CIFAR_LABELS[original_label], f'{idx}_target_{target_label}_achieved_{perturbed_pred}.pt')

    # Extract components. Assume d=1
    H = W = 32
    best_p = perturbations[0]
    # print(f"Best perturbation: {best_p}, type: {type(best_p)}")
    row, col = int(round(best_p[0].item() * H)), int(round(best_p[1].item() * W))
    rgb = best_p[2:].detach().cpu()

    # print(f"Saving perturbation at pixel ({row}, {col}) with RGB values {rgb}")
    payload = {
        'row': row,
        'col': col,
        'rgb': rgb,
    }

    torch.save(payload, perturbed_path)


def main(model_name, n=400, epochs=100, num_images=10, device='cuda'):
    model_path = PRETRAINED_MODELS.get(model_name, None)
    if model_path is None:
        raise ValueError(f"Model {model_name} is not recognized. Available models: {list(PRETRAINED_MODELS.keys())}")
    # Load the model
    model = load_model(model_name, device)
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model: {model_name} from {model_path}")
    print(f"Model architecture: {model}")

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
    samples = get_valid_images(TEST_SET, model_name, device, num_images=num_images)
    print(f"Selected {len(samples)} valid images for the attack.")

    for img, label, idx in samples:
        print(f"\n--- Attacking image {idx} (True Label: {label}) ---")

        # Save the original image
        # save_original_image(img, label, model_name, idx)

        # original_pred = torch.argmax(model(img.unsqueeze(0).to(device)), dim=1).item()

        for target_label in range(10):
            if target_label != label:
                print(f"Targeting class {target_label} ({CIFAR_LABELS[target_label]}) to change from class {label} ({CIFAR_LABELS[label]})")

                # Run attack
                # start = time.perf_counter()
                attack = OnePixelAttack(model, img, label, target_label, n=n)
                perturbed_img, perturbations = attack.perturb_img(epochs=epochs, d=1, show=False, print_every=20)

                # attack_time = time.perf_counter()
                # print(f"Attack time: {attack_time - start:.2f} seconds")

                # Verify attack success
                with torch.no_grad():
                    perturbed_pred = torch.argmax(model(normalize_cifar10(perturbed_img).unsqueeze(0).to(device)), dim=1).item()

                # verify_time = time.perf_counter()
                # print(f"Verification time: {verify_time - attack_time:.2f} seconds")

                print(f"Original prediction: {label}, Perturbed prediction: {perturbed_pred}")

                # Visualize perturbations
                visualize_perturbations(
                    perturbed_img, img, label, model,
                    model_name=model_name,
                    idx=idx,
                    target_label=target_label,
                    perturbed_label=perturbed_pred
                )

                # print(f"Final perturbation: {perturbations[0]}") 

                # Save perturbation
                save_perturbation(perturbations, model_name, idx, label, target_label, perturbed_pred)

                # visualize_time = time.perf_counter()
                # print(f"Visualization time: {visualize_time - verify_time:.2f} seconds")
                # print(f"Total time: {visualize_time - start:.2f} seconds")


if __name__ == "__main__":
    # Example usage
    model_name = 'nin'  # Choose from 'nin', 'conv_allconv', 'original_allconv', 'conv_vgg16', 'original_vgg16'
    # model_path = f'./results/allconv.pth'
    n = 400  # Initial population size for DE
    # batch_size = 100
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Using device: {device}")
    main(model_name, n=n, epochs=epochs, num_images=1, device=device)
