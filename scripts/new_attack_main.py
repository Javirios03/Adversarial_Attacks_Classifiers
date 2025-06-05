import torch
import torchattacks as ta
from utils.attack_aux_funcs import normalize_cifar10, visualize_perturbations, load_model, get_valid_images
from config import IMAGES, MODELS_DICT, PRETRAINED_MODELS, TEST_SET, CIFAR_LABELS
from torchvision.utils import save_image
import time
from models.NormalizedCIFAR10Model import NormalizedCIFAR10Model
import gc


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

    samples = get_valid_images(TEST_SET, model_name, device, num_images=num_images)
    print(f"Selected {len(samples)} valid images for the attack.")

    # Obtain the arrays of images, labels, and indices
    image, _, idx = samples[0]
    image = image.unsqueeze(0).to(device)

    # labels = torch.tensor(labels).to(device)
    # idxs = torch.tensor(idxs)
    # print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    # print(f"Images shape: {images.shape}")

    # Obtain original predictions
    with torch.no_grad():
        original_pred = model(image).argmax(dim=1).item()
    
    print(f"\nImage {idx}: Original label/prediction: {CIFAR_LABELS[original_pred]}")

    adv_images = []
    adv_preds = []

    for label in range(10):
        if label == original_pred:
            continue
    
        print(f"Trying to change to class {CIFAR_LABELS[label]} ({label})")
        target_label = torch.tensor([label]).to(device)
        start = time.time()
        adv_image = attack(image, target_label)  # [N, C, H, W], in our case [N, 3, 32, 32] with N = num_images
        adv_images.append(adv_image.cpu())

        with torch.no_grad():
            pred = model(adv_image).argmax(dim=1).item()

        adv_preds.append(pred)

        # Clean up
        del adv_image
        torch.cuda.empty_cache()
        gc.collect()

        success = (pred == label)
        print(f"Done in {time.time() - start:.2f}s | Predicted: {CIFAR_LABELS[pred]} ({pred}) | Success: {success}")

    print(f"\nSummary for image {idx}:")
    for label, adv_image, pred in zip([i for i in range(10) if i != original_pred], adv_images, adv_preds):
        success = (pred == label)
        print(f"Target: {CIFAR_LABELS[label]} ({label}) | Predicted: {CIFAR_LABELS[pred]} ({pred}) | Success: {success}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_name = 'nin'  # Example model name, can be changed to 'conv_allconv', 'original_allconv', etc.
    main_targeted(model_name=model_name, device=device, num_images=1, d=1, epochs=100, n=400)
