"""
Project: AIGreenBots Project
Author: Victor Romero-Bautista (Docotoral Candidate DC07)
E-Mail: vIcTor.romErobauT@unex.es |victor.romero@inaoep.mx | victor.romerobaut@gmail.com
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
from src.world_model.wm.encoder import Encoder
from src.world_model.wm.decoder import DecoderRGB
from src.world_model.wm.prior import Prior


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = Encoder(img_h=cfg.CAMERA_PARAMETERS.HEIGHT,
                                img_w=cfg.CAMERA_PARAMETERS.WIDTH,
                                latent_dim=cfg.MODEL.LATENT_DIM,
                                action_dim=cfg.MODEL.ACTION_DIM)
        
        self.dec_rgb  = DecoderRGB(img_h=cfg.CAMERA_PARAMETERS.HEIGHT,
                                   img_w=cfg.CAMERA_PARAMETERS.WIDTH,
                                   latent_dim=cfg.MODEL.LATENT_DIM,
                                   reverse_filters=[512, 256, 128, 64])
        
        self.prior    = Prior(latent_dim=cfg.MODEL.LATENT_DIM,
                              action_dim=cfg.MODEL.ACTION_DIM,
                              hidden_dim=cfg.MODEL.HIDDEN_DIM)
        
        self.action_normalizer = ActionNormalizer()

    def encode(self, o_rgb, z_prev, a_prev):
        a_prev_norm = self.action_normalizer.normalize(a_prev)
        mu, log_sig = self.encoder(o_rgb, z_prev, a_prev_norm)
        z = self.encoder.reparam(mu, log_sig)
        return mu, log_sig, z

    def decode(self, z):
        rgb   = self.dec_rgb(z)
        return rgb 

    def predict_prior(self, z_prev, a_prev, h_state=None):
        a_prev_norm = self.action_normalizer.normalize(a_prev)
        return self.prior(z_prev, a_prev_norm, h_state)
    
    def vfe(self, o_rgb, z_mu, z_log_sig, z_prior_mu, z_prior_log_sig, z_sample):
        """
        """
        FREE_BITS = 5.0 
        
        kl_prior = 0.5 * torch.sum(torch.exp(2*(z_log_sig - z_prior_log_sig)) 
                                    + (z_mu - z_prior_mu)**2 / torch.exp(2*z_prior_log_sig) - 1.0 
                                    + 2 * (z_prior_log_sig - z_log_sig),
                                    dim=1).mean()
        kl_prior = torch.max(kl_prior,
                             torch.tensor(FREE_BITS, device=kl_prior.device))
        
        kl_std = 0.5 * torch.sum(torch.exp(2*z_log_sig) + z_mu**2 - 1.0 - 2*z_log_sig, dim=1).mean()
        
        o_rgb_hat = self.decode(z_sample)
        rec_rgb = 0.5 * ((o_rgb - o_rgb_hat)**2).sum([1,2,3]).mean()
        
        beta = 0.5 
        loss = kl_prior + beta * kl_std + rec_rgb 
        return loss, {'kl':      kl_prior.item(),   
                      'kl_std':  kl_std.item(),    
                      'rec_rgb': rec_rgb.item(),   
                      'total':   loss.item(),}     

class ActionNormalizer:
    def __init__(self, v_scale=70.0, delta_scale=0.50):
        self.v_scale = v_scale           
        self.delta_scale = delta_scale   
    
    def normalize(self, action):
        v_norm = action[:, 0] / self.v_scale          
        delta_norm = action[:, 1] / self.delta_scale  
        return torch.stack([v_norm, delta_norm], dim=1)
    
    def denormalize(self, action_norm):
        v = action_norm[:, 0] * self.v_scale
        delta = action_norm[:, 1] * self.delta_scale
        return torch.stack([v, delta], dim=1)
