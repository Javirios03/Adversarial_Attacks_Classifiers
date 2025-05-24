from models.AllConv import AllConv
from models.NiN import NiN
from models.VGG16 import VGG16
import torch
import os


unnormalized_models = {
    "VGG16": ["./results/vgg16_original_acc.pth"],
    "NiN": ["./results/nin_original_acc.pth"],
    "AllConv": ["./results/allconv_original_acc.pth"]
}

for model_name, model_paths in unnormalized_models.items():
    for model_path in model_paths:
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')

        # Check if the checkpoint contains 'model_state'
        if 'model_state' in checkpoint:
            model = checkpoint['model_state']
        else:
            model = checkpoint

        # Save the model with normalized weights
        new_model_path = os.path.join("./results", f"{model_name}_normalized.pth")
        torch.save(model, new_model_path)
        print(f"Saved normalized model to {new_model_path}")