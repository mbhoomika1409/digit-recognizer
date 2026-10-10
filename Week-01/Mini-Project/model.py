import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 64 * 7 * 7 = 3136 numbers
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),         # 10 digits
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = DigitCNN()
    dummy = torch.randn(4, 1, 28, 28)   # 4 fake images
    out = model(dummy)
    print("Output shape:", out.shape)   # should be [4, 10]
    print("Parameters:", sum(p.numel() for p in model.parameters()))