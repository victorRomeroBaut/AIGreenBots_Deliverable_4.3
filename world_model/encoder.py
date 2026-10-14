"""

"""

import torch 
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, action_dim=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,   8, 3, stride=2, padding=1), nn.LeakyReLU(0.2),  # 32×32
            nn.Conv2d(8,  16, 3, stride=2, padding=1), nn.LeakyReLU(0.2),  # 16×16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.2),  #  8×8
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2),  #  4×4
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.LeakyReLU(0.2),  #  2×2
        )
        
        mlp_in = 2 * 2 * 128 + latent_dim + action_dim 
        self.mlp_mu  = nn.Linear(mlp_in, latent_dim)
        self.mlp_sig = nn.Linear(mlp_in, latent_dim)

    def forward(self, x):
        feat = self.cnn(x).flatten(1) 
        return self.mlp_mu(feat), self.mlp_sig(feat)
