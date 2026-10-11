import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, pool: bool = True, stride: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class KWSNet(nn.Module):
    def __init__(self, n_classes: int = 12, dropout: float = 0.3, strided_stem: bool = False):
        super().__init__()
        # strided_stem downsamples in the first block with a stride-2 conv instead
        # of conv + max-pool. Same parameters and output size, but the block no
        # longer computes a full-resolution 16x40x101 map, which was ~2/3 of the
        # forward time on CPU (reports/profiling/README.md).
        stem = (
            ConvBlock(1, 16, pool=False, stride=2) if strided_stem else ConvBlock(1, 16)
        )
        self.features = nn.Sequential(
            stem,
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = KWSNet()
    dummy = torch.randn(8, 1, 40, 101)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"output shape: {tuple(out.shape)}")
    print(f"trainable parameters: {n_params:,}")