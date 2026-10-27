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
import numpy as np
FORWARD_SPEED = 70.00 
STEERING = 0.5
POLICIES = torch.tensor([[FORWARD_SPEED,     0.00],
                        [FORWARD_SPEED, -STEERING],
                        [FORWARD_SPEED, +STEERING],  
                        ], dtype=torch.float32)

POLICY_LABELS = ["Forward",
                "Left",
                "Right"]