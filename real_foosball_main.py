import torch
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from rl_games.algos_torch import torch_ext

import hydra
from omegaconf import DictConfig
import numpy as np

from utilities.models import model_builder
import os
import datetime
from os import path
import sys
import time


keeper_W_Rev_Drive = {
    "name": "Keeper_W_Rev_Drive",
    "eds_location": "C5-E-2-09.eds",
    "node_id": 2,
    "command_limits": [-3585, 3585],
    "range_of_motion": [-2*np.pi, 2*np.pi]
}

keeper_W_Pris_Drive = {
    "name": "Keeper_W_Pris_Drive",
    "eds_location": "C5-E-2-09.eds",
    "node_id": 1,
    "command_limits": [-12150, -200],
    "range_of_motion": [-0.12, 0.12]
}


settings = {
    "device": 'cuda:0',
    "motor_info": [keeper_W_Pris_Drive, keeper_W_Rev_Drive],
    "controlled_rods": ['Keeper_W'],
    "player_rods": ['Keeper_B'],
    "passive_rods": [],
    "observe_joints": True,  # if True, joint positions are used as observations
    "observation_order": ['Keeper_W_Pris_Drive', 'Keeper_W_Rev_Drive', 'Keeper_B', 'Ball'],  # 'Keeper_W_Rev_Drive',
    "detection_model": 'Yolo_Parameters_v0s.pt',
    "detection_classes": [0, 1],  # 0 for figures, 1 for ball
    "prediction_steps": 1
}


class SystemInterface:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict
        self.cfg.test = True
        self.cfg_dict["task"]["test"] = self.cfg.test

    def _get_env_interface(self):
        # First create system interface to get env info
        sys.path.append(path.abspath('../FoosballConnection'))
        from system_interface import FoosballInterface

        settings["device"] = self.cfg["rl_device"]
        self.env_interface = FoosballInterface(settings)

    def _prepare_model_config(self):
        cfg_params = self.rlg_config_dict['config']
        config = {
            'actions_num': self.env_interface.num_motors,
            'input_shape': (self.env_interface.num_obs,),
            'value_size': 1,
            'normalize_value': cfg_params.get('normalize_value', False),
            'normalize_input': cfg_params.get('normalize_input', False)
        }
        return config

    def _get_network_model(self):
        builder = model_builder.CustomModelBuilder()
        model = builder.load(self.rlg_config_dict)
        config = self._prepare_model_config()
        self.model = model.build(config)
        self.model.to(self.cfg["rl_device"])
        self.model.eval()

        # Load trained model parameters
        checkpoint = self.cfg.checkpoint
        if checkpoint is not None and checkpoint != '':
            checkpoint = torch_ext.load_checkpoint(checkpoint)
            self.model.load_state_dict(checkpoint['model'])
            if config['normalize_input'] and 'running_mean_std' in checkpoint:
                self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            return True
        else:
            print("No valid checkpoint has been given. Aborting execution for safety!")
            return False

    # def compute_actions(self, state):
    #     # State is in (n, 1) so transpose
    #     pris = -state[0:1]
    #     ball = state[-4:]
    #     state = torch.concatenate(
    #         (pris, ball)
    #     )
    #     input_dict = {
    #         'obs': state.T.to(torch.float32),
    #         'is_train': False
    #     }
    #     with torch.no_grad():
    #         actions = -self.model(input_dict)['mus'][0]
    #     return actions

    def compute_actions(self, state):
        # State is in (n, 1) so transpose
        kw_pos_t = state[0:1]
        kw_pos_r = state[1:2]
        kb_pos = state[2:3]
        kw_vel_t = state[3:4]
        kw_vel_r = state[4:5]
        kb_vel = state[5:6]
        ball = state[-4:]
        state = torch.concatenate(
            (kw_pos_t, kw_pos_r, kw_vel_t, kw_vel_r, kb_pos, kb_vel, ball)
        )

        # joint_pos = state[:5]
        # ball = state[-4:]
        # state = torch.concatenate((joint_pos, ball))
        input_dict = {
            'obs': self.model.norm_obs(state.T.to(torch.float32)),
            'is_train': False
        }
        self.states.append(state)
        with torch.no_grad():
            mu, _, _, _ = self.model.a2c_network(input_dict)
            # actions = self.model(input_dict)['mus'][0]
            actions = mu[0]
        return actions

    def run(self):
        # create runner and set the settings
        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)['params']
        self._get_env_interface()
        result = self._get_network_model()

        # Shutdowm if checkpoint is invalid for safety
        if not result:
            return

        self.states = []
        self.env_interface.run2(self.compute_actions)
        self.analyse(self.states)

    @staticmethod
    def analyse(states):
        v = torch.stack(states).cpu().numpy()
        print(f"State max = \n{np.max(v, axis=0)}")
        print(f"State min = \n{np.min(v, axis=0)}")
        print(f"State mean = \n{np.mean(v, axis=0)}")
        print("Done.")


@hydra.main(config_name="real_system_config", config_path="cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg.device_id = 0
    cfg.rl_device = f'cuda:{0}'

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        base_path = path.dirname(path.abspath(__file__))
        file_path = path.join(base_path, cfg.checkpoint)
        if path.exists(file_path):
            cfg.checkpoint = file_path
        else:
            cfg.checkpoint = ''

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    rlg_trainer = SystemInterface(cfg, cfg_dict)
    rlg_trainer.run()


if __name__ == '__main__':
    parse_hydra_configs()
