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
import numpy as np
from src.world_model.utils.rgb_utils import preprocess_rgb
from src.world_model.agent.policies import POLICIES, POLICY_LABELS



class AIFAgent:
    def __init__(self, world_model, cfg, device):
        self.wm  = world_model.to(device)
        self.wm.eval()
        self.cfg = cfg
        self.device = device
        self.z_prev = torch.zeros(1, cfg.MODEL.LATENT_DIM).to(device)
        self.a_prev = torch.zeros(1, cfg.MODEL.ACTION_DIM).to(device)
        self.h_lstm = None
        self.z_pref = None
        self.policies = POLICIES.to(cfg.SYSTEM.DEVICE)

    def set_preferred(self, state: np.ndarray):
        rgb_t = preprocess_rgb(state, self.cfg).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _, _ = self.wm.encode(rgb_t,
                                      torch.zeros(1, self.cfg.MODEL.LATENT_DIM, device=self.device),
                                      torch.zeros(1, self.cfg.MODEL.ACTION_DIM,  device=self.device))
        self.z_pref = mu
    
    @torch.no_grad()
    def perceive(self, o_rgb: torch.Tensor):
        mu, log_sig, z = self.wm.encode(o_rgb.to(self.device),
                                        self.z_prev,
                                        self.a_prev)
        self.z_prev = z
        return mu, log_sig, z
    
    @torch.no_grad()
    def compute_efe(self, z_current: torch.Tensor, debug=False) -> tuple:
        if self.z_pref is None:
            efe = np.random.randn(len(self.policies))
            best_k = int(efe.argmin().item())
            return efe, best_k
        
        sigma2    = self.cfg.EFE.SIGMA_PREF ** 2
        gamma     = self.cfg.EFE.GAMMA
        horizon   = self.cfg.EFE.HORIZON
        efe       = np.zeros(len(self.policies))
        epi_total = np.zeros(len(self.policies))
        ext_total = np.zeros(len(self.policies))
        
        for k, policy in enumerate(self.policies):
            a_k     = policy.unsqueeze(0)
            z_hat   = z_current.clone()
            h_state = None
            g_k     = 0.0
            epi_sum = 0.0
            ext_sum = 0.0
            for tau in range(horizon):
                mu_prior, log_sig_prior, h_state = self.wm.predict_prior(z_hat, a_k, h_state)
                z_hat = mu_prior 
                epistemic = torch.mean(log_sig_prior).item() 
                extrinsic = ((z_hat - self.z_pref)**2).mean().item()/(2*sigma2)
                epi_sum += epistemic
                ext_sum += extrinsic
                g_k     += -epistemic + gamma * extrinsic
            efe[k] = g_k
            epi_total[k] = epi_sum
            ext_total[k] = ext_sum

        best_k = int(efe.argmin().item())
        return efe, best_k, epi_total, ext_total

    @torch.no_grad()
    def act(self, bgr: np.ndarray, debug: bool = False) -> tuple:
                
        o_rgb = preprocess_rgb(bgr, self.cfg).unsqueeze(0).to(self.device) 
        z_mu, _, _ = self.perceive(o_rgb) 
        efe_vals, best_k, epi_total, ext_total = self.compute_efe(z_mu, debug=debug)
        action = self.policies[best_k].cpu().numpy()
        v, delta = float(action[0]), float(action[1])
        self.a_prev = self.policies[best_k].unsqueeze(0)
        return v, delta, {"policy_id": POLICY_LABELS[best_k],
                          "gamma":     self.cfg.EFE.GAMMA,
                          "efe_vals":  efe_vals,
                          "epi_total": epi_total,
                          "ext_total": ext_total} 
    
    def preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        rgb_t = preprocess_rgb(rgb, self.cfg)
        return rgb_t 