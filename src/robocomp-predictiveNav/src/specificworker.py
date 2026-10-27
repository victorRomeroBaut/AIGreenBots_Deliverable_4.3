#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2026 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

#from PySide6.QtCore import QTimer
#from PySide6.QtWidgets import QApplication
from glob import glob
import sys, os, time
from rich.console import Console

import cv2 as cv
import numpy as np
import torch
#sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.world_model.config import get_cfg
from src.world_model.wm.world_model import WorldModel
from src.world_model.utils.rgb_utils import preprocess_rgb, make_rgbd_tensor, draw_hud
from src.world_model.agent.aif_agent import AIFAgent

sys.path.append('/opt/robocomp/lib')
from genericworker import *
import interfaces as ifaces
console = Console(highlight=False)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.counter = 0
        self.cfg = get_cfg()
        self.Period = configData["Period"]["Compute"]
        self.operation_mode = self.configData["agent"]["mode"]
        print(self.cfg)
        print(f'mode {self.operation_mode}')
        #? --- Model Agent ---
        self.world_model = self._init_world_model(device=self.cfg.SYSTEM.DEVICE)
        self.agent = AIFAgent(world_model=self.world_model, cfg=self.cfg, device=self.cfg.SYSTEM.DEVICE)
        print("trainer initialized...")
        buf_path = self.configData["agent"]["buffer_path"]
        self._load_buffer(buf_path)
        pref_folder = self.cfg.FOLDER.PREF_STATE
        self._load_preferred_state(pref_folder)
        
        #?- Runtime state
        self.mode = self.configData["agent"]["mode"]
        self.train_every = 50  
        self.step_count = 0
        if self.mode == "deploy":
            self.path_to_results = os.path.join(self.cfg.FOLDER.RESULTS_FOLDER, "deploy_results", time.strftime("%Y%m%d-%H%M%S"))
            self.path_results_img = os.path.join(self.path_to_results, "camera")
            if not os.path.exists(self.path_results_img):
                os.makedirs(self.path_results_img)
            
            with open(os.path.join(self.path_to_results, "actions.txt"), "w") as f:
                f.write(f"time, v, delta, x, z, alpha, Gforward, Gright, Gleft, Policy\n")
            f.close()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes are 'collect' and 'deploy'.")
        
        self._seq = dict(rgb=[], action=[])

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def _init_world_model(self, device: str = "cpu"):
        model = WorldModel(self.cfg)
        checkpoint_path = self.configData["agent"]["checkpoint"]
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(checkpoint)
                model = model.to(device)
                model.eval()
                print(f"World model loaded from {checkpoint_path}, device={device}")
            except Exception as e:
                print(f"Failed to load world model from {checkpoint_path}: {e}")
        else:
            print(f"No world model checkpoint found at {checkpoint_path}. Initializing new model on {device}.")
        return model

    def _load_buffer(self, buffer_path):
        if os.path.exists(buffer_path):
            try:
                self.replay_buffer.load(buffer_path)
                print(f"Replay buffer loaded from {buffer_path}, size={len(self.replay_buffer)} sequences")
            except Exception as e:
                print(f"Failed to load replay buffer from {buffer_path}: {e}")
        else:
            print(f"No replay buffer found at {buffer_path}. Starting with an empty buffer.")

    def __del__(self):
        """Destructor"""


    def _load_preferred_state(self, folder_path):
        import glob
        if not os.path.exists(folder_path):
            print(f"No preferred state folder found at {folder_path}. z_pref will be None.")
            return None
        img_files = glob.glob(os.path.join(folder_path, "img*.jpg"))
        if len(img_files) == 0:
            print(f"No preferred state images found in {folder_path}")
            return None
        print(f"Loading {len(img_files)} preferred state images from {folder_path}")
        z_list = []
        for img_path in img_files:
            img = cv.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}")
                continue
            rgb_t = preprocess_rgb(img, self.cfg).unsqueeze(0).to(self.cfg.SYSTEM.DEVICE)
            with torch.no_grad():
                mu, _, _ = self.world_model.encode(rgb_t,
                                                   torch.zeros(1, self.cfg.MODEL.LATENT_DIM, device=self.cfg.SYSTEM.DEVICE),
                                                   torch.zeros(1, self.cfg.MODEL.ACTION_DIM,  device=self.cfg.SYSTEM.DEVICE))
            z_list.append(mu)
        if len(z_list) == 0:
            print(f"No valid preferred states loaded from {folder_path}")
            return None
        z_pref_avg = torch.stack(z_list).mean(dim=0)
        self.agent.z_pref = z_pref_avg
        torch.save(z_pref_avg.cpu(), os.path.join(folder_path, "z_pref.pth"))
        return True
    
    def test_context_dependence(self):
        import glob
        img_files = glob.glob(os.path.join(self.cfg.FOLDER.PREF_STATE, "img*.jpg"))
    
        if len(img_files) == 0:
            return
    
        img = cv.imread(img_files[0])
        rgb_t = preprocess_rgb(img, self.cfg).unsqueeze(0).to(self.cfg.SYSTEM.DEVICE)
        with torch.no_grad():
            z_zero, _, _ = self.world_model.encode(rgb_t,
                                                   torch.zeros(1, 128, device=self.cfg.SYSTEM.DEVICE),
                                                   torch.zeros(1, 2, device=self.cfg.SYSTEM.DEVICE)
                                                   )
            z_rand, _, _ = self.world_model.encode(rgb_t,
                                                   torch.randn(1, 128, device=self.cfg.SYSTEM.DEVICE),
                                                   torch.tensor([[70.0, 0.0]], device=self.cfg.SYSTEM.DEVICE)
                                                   )
            z_real, _, _ = self.world_model.encode(rgb_t,
                                                   z_zero,
                                                   torch.tensor([[70.0, 0.0]], device=self.cfg.SYSTEM.DEVICE)
                                                   )

    def set_params(self):
        self.mode = self.configData["agent"]["mode"]
        self.period = self.configData["agent"]["period"]
        self.train_every = self.configData["agent"]["train_every"]
        print(f"[worker] mode ={self.mode} period={self.period}ms")
        return True


    @QtCore.Slot()
    def compute(self):
        bgr = self._get_rgb_from_bridge()
        if bgr is not None :
            if self.mode == "collect":
                self._collect_observations(bgr)
            elif self.mode == "deploy":
                self._deploy_step(bgr)
        return True

    def _get_rgb_from_bridge(self):
        try:
            rgb_raw = self.camerargbdsimple_proxy.getImage("camera")
        except Exception as e:
            print(f"Error occurred while fetching RGBD data: {e}")
            return None, None
        w = rgb_raw.width
        h = rgb_raw.height
        rgb = np.frombuffer(rgb_raw.image, dtype=np.uint8).reshape(h, w, 3) #[:,:,::-1] # / 255.0
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)        
        return bgr

    def _deploy_step(self, bgr):
        v, delta, info = self.agent.act(bgr, debug=True)
        try:
            self.omnirobot_proxy.setSpeedBase(0.0, float(v), float(delta)) 
        except Exception as e:
            print(f"Error occurred while setting robot speed: {e}")
        print(f"v=[{v}], delta=[{delta}]")
        rgb_t = self.agent.preprocess(bgr)
        if self.step_count % 10 == 0: 
            rgb_t = preprocess_rgb(bgr, self.cfg).unsqueeze(0).to(self.cfg.SYSTEM.DEVICE)
            with torch.no_grad():
                z_current, _, _ = self.world_model.encode(rgb_t,
                                                          torch.zeros(1, self.cfg.MODEL.LATENT_DIM, device=self.cfg.SYSTEM.DEVICE),
                                                          torch.zeros(1, self.cfg.MODEL.ACTION_DIM,  device=self.cfg.SYSTEM.DEVICE))
            if self.agent.z_pref is not None:
                dist = torch.dist(z_current, self.agent.z_pref).item() #((z_current - self.agent.z_pref)**2).sum().sqrt().item()
                print(f"\n[Step {self.step_count}] Distance to z_pref: {dist:.2f}")
                print(f"  Selected: {info['policy_id']}, v={v:.1f}, delta={delta:.2f}")

                if dist > 100:
                    print(f"WARNING: Large distance to preferred state! Check if the robot is off-track or if z_pref is set correctly.")
                elif dist < 5:
                    print(f"WARNING: Close to preferred state. Good job!")
            else:
                print(f"WARNING: z_pref is None. Cannot compute distance to preferred state.")
            self.test_context_dependence()
        
        
        vis = draw_hud(bgr, info['policy_id'], info['efe_vals'],  info['gamma'])
        cv.imshow('AgriAIF - DEPLOY', vis)
        cv.waitKey(1)
        self.step_count += 1

    def _accumulate_experience(self, rgb_t, action):
        self._seq['rgb'].append(rgb_t) 
        self._seq['action'].append(torch.tensor(action, dtype=torch.float32)) # .cpu()
        
        if len(self._seq['rgb']) == self.cfg.SYSTEM.SEQUENCE_LENGTH:
            print(f"Sequence completed: {len(self._seq['rgb'])} frames. Adding to replay buffer.")
            self.replay_buffer.add_sequence(self._seq['rgb'], self._seq['action'])
            self._seq = dict(rgb=[], action=[])
    
    def _save(self):
        
        self.replay_buffer.save(os.path.join(self.path_to_save_buffer, self.data_name))
        print(f"Model and replay buffer saved to {self.cfg.FOLDER.LOG_FOLDER} and {self.path_to_save_buffer}")


    def startup_check(self):
        print(f"Testing RoboCompCameraRGBDSimple.Point3D from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.Point3D()
        print(f"Testing RoboCompCameraRGBDSimple.TPoints from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TPoints()
        print(f"Testing RoboCompCameraRGBDSimple.TImage from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TImage()
        print(f"Testing RoboCompCameraRGBDSimple.TDepth from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TDepth()
        print(f"Testing RoboCompCameraRGBDSimple.TRGBD from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TRGBD()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        QTimer.singleShot(200, QApplication.instance().quit)





    ######################
    # From the RoboCompCameraRGBDSimple you can call this methods:
    # RoboCompCameraRGBDSimple.TRGBD self.camerargbdsimple_proxy.getAll(str camera)
    # RoboCompCameraRGBDSimple.TDepth self.camerargbdsimple_proxy.getDepth(str camera)
    # RoboCompCameraRGBDSimple.TImage self.camerargbdsimple_proxy.getImage(str camera)
    # RoboCompCameraRGBDSimple.TPoints self.camerargbdsimple_proxy.getPoints(str camera)

    ######################
    # From the RoboCompCameraRGBDSimple you can use this types:
    # ifaces.RoboCompCameraRGBDSimple.Point3D
    # ifaces.RoboCompCameraRGBDSimple.TPoints
    # ifaces.RoboCompCameraRGBDSimple.TImage
    # ifaces.RoboCompCameraRGBDSimple.TDepth
    # ifaces.RoboCompCameraRGBDSimple.TRGBD

    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # RoboCompOmniRobot.void self.omnirobot_proxy.correctOdometer(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.getBasePose(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.getBaseState(RoboCompGenericBase.TBaseState state)
    # RoboCompOmniRobot.void self.omnirobot_proxy.resetOdometer()
    # RoboCompOmniRobot.void self.omnirobot_proxy.setOdometer(RoboCompGenericBase.TBaseState state)
    # RoboCompOmniRobot.void self.omnirobot_proxy.setOdometerPose(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.setSpeedBase(float advx, float advz, float rot)
    # RoboCompOmniRobot.void self.omnirobot_proxy.stopBase()

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # ifaces.RoboCompOmniRobot.TMechParams


