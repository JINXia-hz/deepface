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


class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            block(1, 16),    
            blockpool(16, 16),
            block(16, 32),
            blockpool(32, 32),
            block(32, 64),
            blockpool(64, 64),
            block(64, 128),
        )

    def forward(self, x):
        x = self.features(x)      # [B, 128, 4, 4]
        x = x.mean(dim=(2, 3))    # [B, 128]
        return F.normalize(x, dim=1)

