"""
Project:AIGreenBots Project funded by the European Union's Horizon 2020 
        research and innovation programme under grant agreement No 101169330.
        Individual research project: 
            Cognitive architecture for agricultural robots supporting anticipation.
Author: Victor Romero-Bautista (Doctoral Candidate #07 (DC 07))
        Marie Skłodowska-Curie Actions (MSCA) Fellowship
        Project: AIGreenBots
        E-Mail: victor.romerobaut@unex.es | victor.romerobaut@gmail.com
        Website: https://www.aigreenbots.eu
        Philosophy of Doctor Student
        At:
        Artificial Vision and Robotics Laboratory (RoboLab),
        Escuela Politécnica de Cáceres (EPC),
        Universidad de Extremadura (UEX)

"""

import torch
import torch.nn as nn

class DecoderRGB(nn.Module):
    
    def __init__(self, img_h:int, img_w:int,
                 latent_dim: int = 128,
                 reverse_filters: list = [512, 256, 128, 64]):
        super(DecoderRGB, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.fc = nn.Linear(latent_dim, reverse_filters[0] * img_h//32 * img_w//32) # Project latent vector to a feature map (B, 512*2, h/32, w/32)
        layers = []
        in_channels = reverse_filters[0]
        for i, out_channels in enumerate(reverse_filters[1:]):
            layers += [nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1),
                      nn.LeakyReLU(0.2, inplace=True)]
            in_channels = out_channels
        layers += [nn.ConvTranspose2d(in_channels=reverse_filters[-1],
                                      out_channels=in_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                   nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=3,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                   nn.Sigmoid()
                   ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                                             
        x = x.view(x.size(0), -1, self.img_h//32, self.img_w//32)  
        x = self.deconv(x)                                         
        return x


class DecoderCatal(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8) 

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64,  32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32,  16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16,   8, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(8,    3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 8, 8)
        return self.net(x)

class Decoder_Depth(nn.Module):
    
    def __init__(self,
                 img_h: int, img_w: int,
                 latent_dim: int = 128,
                 reverse_filters: list = [512, 256, 128, 64]):
        super(Decoder_Depth, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.fc = nn.Linear(latent_dim, reverse_filters[0] * img_h//32 * img_w//32) # Project latent vector to a feature map (B, 512*2, 4, 4)
        layers = []
        in_channels = reverse_filters[0]
        for i, out_channels in enumerate(reverse_filters[1:]):
            layers += [nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1),
                      nn.LeakyReLU(0.2, inplace=True)
                      ]
            in_channels = out_channels
        layers += [nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=1,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                   nn.Softplus()
                   ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                                            
        x = x.view(x.size(0), -1, self.img_h//32, self.img_w//32) 
        x = self.deconv(x)                                        
        return x


if __name__ == "__main__":
    pass