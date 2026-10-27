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
FORWARD_SPEED = 70.00 
STEERING = 0.5
POLICIES = torch.tensor([[FORWARD_SPEED,     0.00],
                        [FORWARD_SPEED, -STEERING],
                        [FORWARD_SPEED, +STEERING],  
                        ], dtype=torch.float32)

POLICY_LABELS = ["Forward",
                "Left",
                "Right"]