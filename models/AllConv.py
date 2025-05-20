import torch.nn as nn
import torch


# class AllConv_Draft_1(nn.Module):
#     def __init__(self, num_classes=10):
#         super(AllConv, self).__init__()
#         self.features = nn.Sequential(
#             # Input: 3 x 32 x 32, Depth: 96
#             nn.Conv2d(3, 96, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             # Input: 96 x 32 x 32, Depth: 96
#             nn.Conv2d(96, 96, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             # Input: 96 x 32 x 32, Depth: 96
#             # Strided convolution to substitute pooling
#             nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),

#             # Input: 96 x 16 x 16, Depth: 192
#             nn.Conv2d(96, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             # Input: 192 x 16 x 16, Depth: 192
#             nn.Conv2d(192, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             # Add Dropout layer
#             nn.Dropout(0.3),

#             # Input: 192 x 16 x 16, Depth: 192
#             nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),

#             # Input: 192 x 8 x 8, Depth: 192
#             nn.Conv2d(192, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             # Input: 192 x 8 x 8, Depth: 192
#             nn.Conv2d(192, 192, kernel_size=1),
#             nn.ReLU(inplace=True),

#             # Input: 192 x 8 x 8, Depth: num_classes
#             nn.Conv2d(192, num_classes, kernel_size=1),
#             nn.ReLU(inplace=True),

#             # nn.AvgPool2d(kernel_size=6, stride=1)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         # Flatten the output to (batch_size, num_classes)
#         return torch.flatten(x, 1)

class AllConv(nn.Module):
    def __init__(self, dropout=True, num_classes=10):
        super(AllConv, self).__init__()

        self.p_dropout = 0.3 if dropout else 0.0

        self.features = nn.Sequential(
            # Input: 3 x 32 x 32, Depth: 96
            nn.Conv2d(3, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Input: 96 x 32 x 32, Depth: 96
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Input: 96 x 32 x 32, Depth: 96
            # Strided convolution to substitute pooling
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p_dropout),

            # Input: 96 x 16 x 16, Depth: 192
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Input: 192 x 16 x 16, Depth: 192
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Input: 192 x 16 x 16, Depth: 192
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p_dropout),

            # Input: 192 x 8 x 8, Depth: 192
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Input: 192 x 8 x 8, Depth: 192
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),

            # Input: 192 x 8 x 8, Depth: num_classes
            nn.Conv2d(192, num_classes, kernel_size=1),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)  # [batch_size, num_classes, 1, 1]
        return x.view(x.size(0), -1)  # Flatten the output to (batch_size, num_classes)
