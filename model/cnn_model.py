import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicCNN(nn.Module):
    def __init__(self, num_classes):
        super(MusicCNN, self).__init__()

        # 🎵 Convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/8
        )

        # 🔥 Adaptive pooling → removes need to calculate size manually
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 🧠 Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


# 🔥 Optional: quick test (VERY useful)
if __name__ == "__main__":
    model = MusicCNN(num_classes=10)

    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)

    print("Output shape:", output.shape)