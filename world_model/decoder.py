"""
A
"""

import torch
import torch.nn as nn


class Decoder_Depth(nn.Module):
    """

    """
    def __init__(self, ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),  # 4×4
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),   # 8×8
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),   # 16×16
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2),    # 32×32
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),           # 64×64
        )

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        """
        
        """
        z = z.view(z.size(0), 128, 1, 1)
        z = self.cnn(z)
        return z

