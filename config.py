from models.AllConv import AllConv, AllConv_K5, AllConv_K7
from models.NiN import NiN
from models.VGG16 import VGG16
from torchvision import transforms, datasets
import torch

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

# Training specific configurations
DATASET = "cifar10"
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.01

# Data transformations - For adversarial attacks, we do not normalize the images
CIFAR_10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
TEST_SET = datasets.CIFAR10(root='./data', train=False, download=False, transform=TEST_TRANSFORM)

# Paths
IMAGES = "./data/images"
MODELS_DICT = {
    'nin': NiN,
    'conv_allconv': AllConv,
    'original_allconv': AllConv,
    'allconv_k5': AllConv_K5,
    'allconv_k7': AllConv_K7,
    'conv_vgg16': VGG16,
    'original_vgg16': VGG16
}
PRETRAINED_MODELS = {
    'nin': './results/nin.pth',
    'conv_allconv': './results/allconv.pth',
    'original_allconv': './results/allconv_original_acc.pth',
    'allconv_k5': './results/allconv_k5_original_acc.pth',
    'allconv_k7': './results/allconv_k7_original_acc.pth',
    'conv_vgg16': './results/vgg16.pth',
    'original_vgg16': './results/vgg16_original_acc.pth'
}

# Logging for attacks
FIELDNAMES = [
    'image_idx',
    'model_name',
    'orig_label',
    'target_label',
    'adv_pred',
    'success',
    'time_per_attack'
]