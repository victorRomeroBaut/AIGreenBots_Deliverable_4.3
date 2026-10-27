"""
Project: AIGreenBots Project
Author: Victor Romero-Bautista (Docotoral Candidate DC07)
E-Mail: victor.romErobauT@unex.es |victor.romero@inaoep.mx | victor.romerobaut@gmail.com
Philosophy of Doctor Student
At:
Artificial Vision and Robotics Laboratory (RoboLab),
Escuela Politécnica de Cáceres (EPC),
Universidad de Extremadura (UEX)
and
Computer Vision Laboratory,
Coordinación de Ciencias Computacionales (CCC),
Instituto Nacional de Astrofísica Óptica y Electrónica (INAOE)

"""

import torch 
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary


class ResNetStem(nn.Module):
    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential( 
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4
        )
        self.proj = nn.Sequential(
            nn.Conv2d(512, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.stem(x))
        return x                   


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu        = nn.Linear(hidden_dim, output_dim) 
        self.log_sigma = nn.Linear(hidden_dim, output_dim) 
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor): 
        x = self.net(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        return mu, log_sigma

class Encoder(nn.Module):
    
    def __init__(self, img_h:int, img_w:int,
                 latent_dim: int = 128,
                 action_dim: int = 2):
        super(Encoder, self).__init__()
        self.cnn_encoder = ResNetStem(out_channels=512, pretrained=True) 
        mlp_in = img_w//32 * img_h//32 * 512 + latent_dim + action_dim
        self.mlp_encoder = MLP(input_dim=mlp_in, hidden_dim=256, output_dim=latent_dim) 

    def forward(self, x:torch.Tensor, z_prev:torch.Tensor, a_prev:torch.Tensor) -> (torch.Tensor, torch.Tensor): 
        
        x = self.cnn_encoder(x)                     
        x = x.view(x.size(0), -1)                   
        h = torch.cat([x, z_prev, a_prev], dim=-1)  
        mu, log_sigma = self.mlp_encoder(h) 
        return mu, log_sigma
    
    #@staticmethod
    def reparam(self, mu:torch.Tensor, log_sigma:torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        z_t = mu + sigma * eps
        return z_t

class EncoderCatal(nn.Module):
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

    def forward(self, x, z_prev, a_prev):
        feat = self.cnn(x).flatten(1)
        h = torch.cat([feat, z_prev, a_prev], dim=-1)
        return self.mlp_mu(h), self.mlp_sig(h)
    
    #@staticmethod
    def reparam(self, mu:torch.Tensor, log_sigma:torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma) 
        eps = torch.randn_like(mu)
        z_t = mu + sigma * eps
        return z_t


if __name__ == "__main__":
    net = Encoder(img_h=256, img_w=256, latent_dim=128, action_dim=2)
    x = torch.zeros(1, 3, 256, 256)
    z_prev = torch.zeros(1, 128)
    a_prev = torch.zeros(1, 2)
    summary(net,input_data=[x, z_prev, a_prev]) 
    out = net(x, z_prev, a_prev)
    print(out[0].shape)