import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time

# Models
from models.AllConv import AllConv
from models.NiN import NiN
from models.VGG16 import VGG16

# Auxiliary scripts
from scripts.OPA_funcs import OnePixelAttack
from utils.attack_aux_funcs import visualize_perturbations, CIFAR_LABELS


def main(model_name, model_path, n=400, batch_size=100, epochs=100, device='cuda'):
    # Load the model
    if model_name == 'AllConv':
        model = AllConv()
    elif model_name == 'NiN':
        model = NiN()
    elif model_name == 'VGG16':
        model = VGG16()
    else:
        raise ValueError("Model not recognized")

    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Load CIFAR-10 dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=test_transform)

    # Obtain a random image and its label
    idx = int(torch.randint(0, len(test_dataset), (1,)).item())
    img, label = test_dataset[idx]  # label is an integer from 0 to 9

    # Convert to tensor and move to device
    img = img.to(device)

    with torch.no_grad():
        original_pred = torch.argmax(model(img.unsqueeze(0).to(device)), dim=1).item()
    # print(model(img.unsqueeze(0).to(device)))  # Raw logits for each class. - Example: tensor([[-6.2348, -5.0969, -2.6802, 20.7050, -6.4521,  9.5229,  1.6183, -2.0581, -7.1109, -8.2980]], grad_fn=<ViewBackward0>)
    # print(F.softmax(model(img.unsqueeze(0).to(device)), dim=1))  # Related probabilities for each class. - Example for previous logits: tensor([[1.9961e-12, 6.2285e-12, 6.9814e-11, 9.9999e-01, 1.6063e-12, 1.3922e-05, 5.1375e-09, 1.3005e-10, 8.3125e-13, 2.5361e-13]], grad_fn=<SoftmaxBackward0>)
    # print(torch.argmax(model(img.unsqueeze(0).to(device)), dim=1))  # Original prediction. - Example: tensor([3])
    print(f"Original prediction: {original_pred}")  # Predicted class: 3
    print(f"Original label: {label}")  # Real label: 3

    for target_label in range(10):
        if target_label != label:
            print(f"Trying to change label {label} to {target_label} with classes {CIFAR_LABELS[label]} to {CIFAR_LABELS[target_label]}")
            # Run attack
            start = time.perf_counter()
            attack = OnePixelAttack(model, img, label, target_label, n=n)
            perturbed_img, _ = attack.perturb_img(epochs=epochs, d=1, show=True, print_every=10, title=f"One Pixel Attack on {model_name} - {n} pixels")

            attack_time = time.perf_counter()
            print(f"Attack time: {attack_time - start:.2f} seconds")

            # Verify attack success
            with torch.no_grad():
                perturbed_pred = torch.argmax(model(perturbed_img.unsqueeze(0).to(device)), dim=1).item()

            verify_time = time.perf_counter()
            print(f"Verification time: {verify_time - attack_time:.2f} seconds")

            print(f"Original prediction: {original_pred}, Perturbed prediction: {perturbed_pred}")

            # Visualize perturbations
            visualize_perturbations(perturbed_img, img, label, model, title=f"OPA - {model_name} - Original {label}-{original_pred} - Tried {target_label}-{perturbed_pred}")

            visualize_time = time.perf_counter()
            print(f"Visualization time: {visualize_time - verify_time:.2f} seconds")
            print(f"Total time: {visualize_time - start:.2f} seconds")


if __name__ == "__main__":
    # Example usage
    model_name = 'AllConv'  # Choose from 'AllConv', 'NiN', 'VGG16'
    model_path = f'./results/allconv.pth'
    n = 400  # Initial population size for DE
    batch_size = 100
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Using device: {device}")
    main(model_name, model_path, n, batch_size, epochs, device)
