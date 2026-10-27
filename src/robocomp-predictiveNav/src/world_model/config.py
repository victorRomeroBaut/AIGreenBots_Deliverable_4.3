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

import os
import yaml
from yacs.config import CfgNode as CN
main_path = os.path.dirname(__file__)

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cpu"
_C.SYSTEM.BATCH_SIZE = 4
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.PIN_MEMORY = False
_C.SYSTEM.BUFFER_SIZE = 50000
_C.SYSTEM.SEQUENCE_LENGTH = 10
_C.DATA = CN()
_C.DATA.RESCALE = True
_C.DATA.NORMALIZE = True
_C.DATA.MEANS = [0.5, 0.5, 0.5]
_C.DATA.STDS = [0.5, 0.5, 0.5]
_C.DATA.IMG_FORMAT = "jpg"
_C.CAMERA_PARAMETERS = CN()
_C.CAMERA_PARAMETERS.HEIGHT = 1
_C.CAMERA_PARAMETERS.WIDTH = 1
_C.OPTIMIZATION = CN()
_C.OPTIMIZATION.OPTIM = "Adam" 
_C.OPTIMIZATION.LR = 5e-4 
_C.OPTIMIZATION.NUM_EPOCHS = 10
_C.LOGIN = CN()
_C.LOGIN.VAL_FREQUENCY = 1
_C.LOGIN.SAVE_FREQUENCY = 5
_C.LOADING = CN()
_C.LOADING.FINE_TUNNING = None
_C.LOADING.NETWORKS_WEIGHTS_FOLDER = None 
_C.LOADING.WEIGHTS_TO_TEST_FOLDER = None
_C.LOADING.EPOCH_TEST = None
_C.FOLDER = CN()
_C.FOLDER.LOG_FOLDER = "checkpoints"
_C.FOLDER.RESULTS_FOLDER = "results"
_C.FOLDER.BUFFER_FOLDER = "buffer"
_C.FOLDER.PREF_STATE = "preferred_state"
_C.MODEL = CN()
_C.MODEL.VERSION = "v1"
_C.MODEL.MODEL_NAME = "CNN_LSTM"
_C.MODEL.HIDDEN_SIZE = 1000
_C.MODEL.INPUT_CHANNELS = 3
_C.MODEL.LATENT_DIM = 128
_C.MODEL.HIDDEN_DIM = 512
_C.MODEL.ACTION_DIM = 2
_C.MODEL.TRAIN = False
_C.MODEL.TEST = False
_C.MODEL.RESUME_TRAINING = False
_C.MODEL.SHOW_TEST_RESULT = False
_C.MODEL.ACTION_V_SCALE = 70.0
_C.MODEL.ACTION_DELTA_SCALE = 0.50
_C.EFE = CN()
_C.EFE.HORIZON = 10   
_C.EFE.N_POLICIES = 7 
_C.EFE.SIGMA_PREF = 0.5
_C.EFE.EMERGENCY_DISTANCE = 0.5
_C.EFE.GAMMA = 1.0
_C.ADAPTIVE_PRECISION = CN()
_C.ADAPTIVE_PRECISION.GAMMA_BASE = 1.0
_C.ADAPTIVE_PRECISION.GAMMA_MAX = 100.0
_C.ADAPTIVE_PRECISION.LAMBDA_D = 3.0

def _update_config_from_file(config, cfg_file):
    config.defrost()
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config): 
    _update_config_from_file(config, os.path.join("/home/victor/Dev/AgriCog/src/robocomp-predictiveNav/src/world_model/configs/", "AIF.yaml")) 
    _update_config_from_file(config, os.path.join("/home/victor/Dev/AgriCog/src/robocomp-predictiveNav/src/world_model/configs/", "Webots_AgriWorld.yaml")) # 


def get_cfg(): 
    config = _C.clone()
    update_config(config)
    return config