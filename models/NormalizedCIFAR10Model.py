import torch
import torch.nn as nn
from config import CIFAR_10_MEAN, CIFAR_10_STD


class NormalizedCIFAR10Model(nn.Module):
    def __init__(self, base_model):
        super(NormalizedCIFAR10Model, self).__init__()
        self.base_model = base_model
        self.register_buffer('mean', CIFAR_10_MEAN.view(1, 3, 1, 1))
        self.register_buffer('std', CIFAR_10_STD.view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model with normalization.

        Parameters:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Output tensor after normalization and passing through the base model.
        """
        # return self.base_model(x)
        return self.base_model((x - self.mean) / self.std)
