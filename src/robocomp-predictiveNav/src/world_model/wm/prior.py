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

class Prior(nn.Module):
    def __init__(self, latent_dim: int = 128,
                 action_dim: int = 2,
                 hidden_dim: int = 512):
        super(Prior, self).__init__()
        lstm_in = latent_dim + action_dim
        self.lstm_prior = nn.LSTM(input_size=lstm_in,
                                  hidden_size=hidden_dim,
                                  batch_first=True)         
        self.mu = nn.Linear(hidden_dim, latent_dim)         
        self.log_sigma = nn.Linear(hidden_dim, latent_dim)  
    
    def forward(self, z_prev:torch.Tensor, a_prev:torch.Tensor, h_state:torch.Tensor=None) -> (torch.Tensor, torch.Tensor, torch.Tensor): # 
        if z_prev.dim() == 2:             
            z_prev = z_prev.unsqueeze(1)  
            a_prev = a_prev.unsqueeze(1)  
            squeezed = True
        else:
            squeezed = False
        
        x = torch.cat([z_prev, a_prev], dim=-1)             
        lstm_out, new_h_state = self.lstm_prior(x, h_state) 
        mu = self.mu(lstm_out)                              
        log_sigma = self.log_sigma(lstm_out).clamp(-4, 4)   
        if squeezed:
            mu = mu.squeeze(1)                              
            log_sigma = log_sigma.squeeze(1)                
        return mu, log_sigma, new_h_state
    