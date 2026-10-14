"""

"""

import torch
import torch.nn as nn

class DVIO(torch.nn.Module):
    def __init__(self, ):
        super(DVIO, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x1) -> torch.Tensor:
        x1 = self.cnn(x1)
        return x1