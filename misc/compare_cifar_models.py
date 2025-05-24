from models.AllConv import AllConv
from models.NiN import NiN
from models.VGG16 import VGG16

from torchsummary import summary
from ptflops import get_model_complexity_info
import time
import torch
import warnings
import os
import contextlib

warnings.filterwarnings("ignore", category=UserWarning)


results_path = "./results/"
models_paths = {
    "VGG16": ["vgg16.pth", "vgg16_original_acc.pth"],
    "NiN": ["nin_original_acc.pth"],
    "AllConv": ["allconv.pth", "allconv_original_acc.pth"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def print_model_info(model, model_name: str, input_size=(3, 32, 32)):
    """
    Print model information including number of parameters and FLOPs.
    """
    model.to(device)
    model.eval()
    print(f"\n{'='*40}")
    print(f"Info for: {model_name}")
    print(f"{'='*40}")

    # 1. Model summary
    try:
        summary(model, input_size, device=str(device))
    except Exception as e:
        print("Error with summary:", e)

    # 2. FLOPs and Params
    with torch.cuda.amp.autocast(enabled=False):
        try:
            macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                                     print_per_layer_stat=False, verbose=False)
            print(f"FLOPs (MACs): {macs}")
            print(f"Params: {params}")
        except Exception as e:
            print("Error with ptflops:", e)

    # 3. Inference time
    dummy_input = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        end = time.time()
        avg_time_ms = (end - start) / 100 * 1000
        print(f"Average Inference Time: {avg_time_ms:.2f} ms")

# Loop through all models and their variants
for arch, paths in models_paths.items():
    for path in paths:
        # Load model
        if arch == "VGG16":
            model = VGG16()
        elif arch == "NiN":
            model = NiN()
        elif arch == "AllConv":
            model = AllConv()
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Load weights
        full_path = os.path.join(results_path, path)
        state_dict = torch.load(full_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:  # Model wasn't saved using only model's state_dict
            model.load_state_dict(state_dict['model_state'])

        # Print model info
        # print_model_info(model, input_size=(3, 32, 32), model_name=f"{arch} - {path}")
        output_file = f"arch_{arch}_{os.path.splitext(path)[0]}_info.txt"
        output_path = os.path.join("models_info", output_file)

        os.makedirs("models_info", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            print_model_info(model, model_name=f"{arch} - {path}", input_size=(3, 32, 32))
