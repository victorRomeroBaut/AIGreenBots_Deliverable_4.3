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

import numpy as np
import cv2 as cv
import torch

def preprocess_rgb(frame_bgr: np.ndarray, cfg) -> torch.Tensor:
    """
    """
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    frame_resized = cv.resize(frame_rgb,
                              (cfg.CAMERA_PARAMETERS.WIDTH,
                               cfg.CAMERA_PARAMETERS.HEIGHT),
                              interpolation=cv.INTER_LINEAR)
    tensor = torch.from_numpy(frame_resized).float() / 255.0
    return tensor.permute(2, 0, 1)


def make_rgbd_tensor(rgb_tensor: torch.Tensor,
                     depth_tensor: torch.Tensor) -> torch.Tensor:
    
    return torch.cat([rgb_tensor, depth_tensor], dim=0)


def augment_rgb(rgb_tensor: torch.Tensor, cfg) -> torch.Tensor:
    img = rgb_tensor.numpy().transpose(1, 2, 0)  
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv.cvtColor(img_uint8, cv.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= np.random.uniform(0.7, 1.3)  
    hsv[..., 2] *= np.random.uniform(0.6, 1.4)  
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img_aug = cv.cvtColor(hsv, cv.COLOR_HSV2RGB).astype(np.float32) / 255.0
    if np.random.random() < 0.3:
        row = np.random.randint(0, cfg.CAMERA_PARAMETERS.HEIGHT - 8)
        h   = np.random.randint(2, 8)
        img_aug[row:row+h, :, :] = 0.0
    if np.random.random() < 0.2:
        sigma = np.random.uniform(0.5, 2.0)
        img_aug = cv.GaussianBlur(img_aug, (0, 0), sigma)

    tensor_aug = torch.from_numpy(img_aug).float().permute(2, 0, 1)
    return tensor_aug.clamp(0.0, 1.0)


def visualise_imagination(imagined_rgb_list: list,
                          policy_labels: list) -> np.ndarray:
    
    rows = []
    for k, frames in enumerate(imagined_rgb_list):
        label = policy_labels[k]
        row_frames = []
        for f in frames:
            img = (f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img = cv.resize(img, (80, 64))
            row_frames.append(img)
        row = np.concatenate(row_frames, axis=1)
        cv.putText(row, label, (5, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        rows.append(row)
    return np.concatenate(rows, axis=0)

def draw_hud(bgr: np.ndarray, policy_label: str,
             efe_vals, gamma: float) -> np.ndarray:
    vis = bgr.copy()
    cv.putText(vis, f"Policy: {policy_label}", (5, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv.putText(vis, f"Gamma : {gamma}", (5, 38),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if efe_vals is not None:
        from src.world_model.agent.policies import POLICY_LABELS
        for k, g in enumerate(efe_vals):
            cv.putText(vis, f"G[{k}] {POLICY_LABELS[k][:6]}={g:.1f}",
                        (5, 62 + k * 14),
                        cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)
    return vis