import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


def blockpool(cin, cout):
    return nn.Sequential(
        block(cin, cout),
        nn.MaxPool2d(2),
    )


class Net(nn.Module):
    """36x36 grayscale window classifier (heatmap backend)."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            block(1, 16),
            blockpool(16, 16),
            block(16, 32),
            blockpool(32, 32),
            block(32, 64),
            blockpool(64, 64),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)      # [B, 64, 4, 4]
        x = x.mean(dim=(2, 3))    # [B, 64]
        return self.fc(x)


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU"""

    def __init__(self, cin, cout, kernel_size=3):
        super().__init__(
            nn.Conv2d(
                cin,
                cout,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class Block(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()

        self.conv1 = ConvBNReLU(cin, cout)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                cout,
                cout,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
        )

        if cin == cout:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class BlockPool(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class BlockUp(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.fuse = Block(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)

        return x


class ResUNet(nn.Module):
    """Dense per-pixel face score map (deep backend, stride 1)."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=32,
        num_blocks=2,
    ):
        super().__init__()

        b = base_channels

        self.pool1 = BlockPool(in_channels, b)
        self.enc1 = nn.Sequential(
            *[Block(b, b) for _ in range(num_blocks)]
        )

        self.pool2 = BlockPool(b, b * 2)

        self.bottleneck = nn.Sequential(
            *[Block(b * 2, b * 2) for _ in range(num_blocks)]
        )

        self.up2 = BlockUp(
            in_channels=b * 2,
            skip_channels=b * 2,
            out_channels=b * 2,
        )
        self.dec2 = nn.Sequential(
            *[Block(b * 2, b * 2) for _ in range(num_blocks)]
        )

        self.up1 = BlockUp(
            in_channels=b * 2,
            skip_channels=b,
            out_channels=b,
        )
        self.dec1 = nn.Sequential(
            *[Block(b, b) for _ in range(num_blocks)]
        )

        self.head = nn.Conv2d(
            b,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        x, skip1 = self.pool1(x)
        x = self.enc1(x)

        x, skip2 = self.pool2(x)

        x = self.bottleneck(x)

        x = self.up2(x, skip2)
        x = self.dec2(x)

        x = self.up1(x, skip1)
        x = self.dec1(x)

        x = self.head(x)

        return x


def load_model(path):
    """Load a state dict into the right architecture; returns (net, backend).

    backend is 'heatmap' for Net (stride-8 window scores) or 'deep' for
    ResUNet (stride-1 dense score map), inferred from the state dict keys.
    """
    sd = torch.load(path, weights_only=True)
    if any(k.startswith('pool1.') for k in sd):
        net, backend = ResUNet(in_channels=1, out_channels=1), 'deep'
    else:
        net, backend = Net(), 'heatmap'
    net.load_state_dict(sd)
    net.eval()
    return net, backend
