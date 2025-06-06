import torch
import torchattacks as ta
from utils.attack_aux_funcs import normalize_cifar10, visualize_perturbations, load_model, get_valid_images, set_seed
from config import IMAGES, MODELS_DICT, PRETRAINED_MODELS, TEST_SET, CIFAR_LABELS, FIELDNAMES
from torchvision.utils import save_image
import time
from models.NormalizedCIFAR10Model import NormalizedCIFAR10Model
import gc
import csv
from pathlib import Path


def main_not_targeted(model_name: str, d=1, epochs=100, n=400, device='cuda', num_images=10):
    base_model = load_model(model_name, device)
    model = NormalizedCIFAR10Model(base_model).to(device)
    model.eval()
    # No need to specify F or CR since ta.OnePixel is a direct implementation of the paper's method
    attack = ta.OnePixel(
        model,
        pixels=d,
        steps=epochs,
        popsize=n
    )
    attack.set_mode_default()  # Ensure not targeted mode
    # attack._set_normalization_applied(False)  # Tell the algorithm to apply normalization itself (inputs in [0, 1] range)

    samples = get_valid_images(TEST_SET, model_name, device, num_images=num_images)
    print(f"Selected {len(samples)} valid images for the attack.")

    # Obtain the arrays of images, labels, and indices
    images, labels, idxs = zip(*samples)

    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)
    # idxs = torch.tensor(idxs)
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    # Obtain original predictions
    with torch.no_grad():
        original_preds = model(images).argmax(dim=1)

    start = time.time()
    adv_images = attack(images, labels)  # [N, C, H, W], in our case [N, 3, 32, 32] with N = num_images
    print(f"Adversarial attack completed in {time.time() - start:.2f} seconds.")

    # Obtain adversarial predictions
    with torch.no_grad():
        adv_preds = model(adv_images).argmax(dim=1)

    # Check success rate
    success_mask = (adv_preds != labels)
    success_idxs = torch.nonzero(success_mask).squeeze(1)

    print(type(adv_images))
    print(adv_images.shape)
    print(type(adv_images[0]))
    print(adv_images[0].shape)
    print(len(adv_images))

    for i in range(images.size(0)):
        success = bool(success_mask[i].item())
        print(f"Image {idxs[i]}: Original label: {CIFAR_LABELS[labels[i]]}, "
              f"Original prediction: {CIFAR_LABELS[original_preds[i]]}, "
              f"Adversarial prediction: {CIFAR_LABELS[adv_preds[i]]}, "
              f"Success: {success}")

    # # Save adversarial images
    # # for i, (adv_img, label, idx) in enumerate(zip(adv_images, labels, idxs)):        
    # #     save_image(adv_img.cpu(), f"{IMAGES}/{model_name}/{CIFAR_LABELS[label]}/{idx}_adv.png")


def main_targeted(model_name: str, d=1, epochs=100, n=400, device='cuda', num_images=10):
    """
    Perform a targeted adversarial attack on CIFAR-10 images using the OnePixel attack. Restricted, for now, to one image (9 attacks therefore)"""
    base_model = load_model(model_name, device)
    model = NormalizedCIFAR10Model(base_model).to(device)
    model.eval()
    # No need to specify F or CR since ta.OnePixel is a direct implementation of the paper's method
    attack = ta.OnePixel(
        model,
        pixels=d,
        steps=epochs,
        popsize=n
    )
    attack.set_mode_targeted_by_label()  # Ensure targeted mode
    # The desired target is set in the following function call in the labels argument
    # attack._set_normalization_applied(False)  # Tell the algorithm to apply normalization itself (inputs in [0, 1] range) - Necessary?

    samples = get_valid_images(TEST_SET, model_name, device, num_images=num_images)  # For testing purposes, we can force a specific image index (integer)
    print(f"Selected {len(samples)} valid images for the attack.")

    attack_tasks = []
    for img, orig_label, idx in samples:
        for target_label in range(10):
            if target_label == orig_label:
                continue
            attack_tasks.append((img, orig_label, target_label, idx))

    print(f"Total attack tasks: {len(attack_tasks)}")

    # Setup Logging
    log_path = Path(f'logs/{model_name}_targeted_attack_log.csv')
    log_path.parent.mkdir(exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        # Run in batches
        batch_size = 32
        for i in range(0, len(attack_tasks), batch_size):
            batch = attack_tasks[i:i + batch_size]
            
            # Prepare batch tensors
            images = torch.stack([item[0] for item in batch]).to(device)
            orig_labels = [item[1] for item in batch]
            target_labels = torch.tensor([item[2] for item in batch], dtype=torch.long).to(device)
            indices = [item[3] for item in batch]

            print(f"Processing batch {i // batch_size + 1}/{(len(attack_tasks) + batch_size - 1) // batch_size} |Size: {len(batch)}")

            # Perform the attack
            start = time.time()
            adv_images = attack(images, target_labels)
            elapsed = time.time() - start

            # Predict with adversarial images
            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1).cpu().tolist()

            # Log results
            logs = []
            for j in range(len(batch)):
                logs.append({
                    'image_idx': indices[j],
                    'model_name': model_name,
                    'orig_label': CIFAR_LABELS[orig_labels[j]],
                    'target_label': CIFAR_LABELS[target_labels[j].item()],
                    'adv_pred': CIFAR_LABELS[preds[j]],
                    'success': preds[j] == target_labels[j].item(),
                    'time_per_attack': round(elapsed / len(batch), 4)  # Average time per image in the batch
                })
            writer.writerows(logs)

            # Clean up
            del adv_images
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    # set_seed(42)  # Set a random seed for reproducibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_name = 'nin'  # Example model name, can be changed to 'conv_allconv', 'original_allconv', etc.
    main_targeted(model_name=model_name, device=device, num_images=10, d=1, epochs=100, n=400)
