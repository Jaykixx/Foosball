from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from rl_games.algos_torch import torch_ext

import hydra
from omegaconf import DictConfig
from gym import spaces
import numpy as np

from utilities.custom_runner import CustomRunner as Runner
from utilities.models import model_builder
import os
import datetime
from os import path
import sys


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
        # TODO: We probably need to install motor402 to run
        answer = None
        while not ((answer == 'y') or (answer == 'n')):
            answer = input(
                "Please check if motor limits are correctly set before continuing. "
                "Do you want to continue? y/n"
            )
        if answer == 'n':
            return

        # TODO: Send player model to Interface? Or go different route?
        self.env_interface = FoosballInterface()

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
        self.model = model.build(config).a2c_network

        # Load trained model parameters
        checkpoint = self.cfg.checkpoint
        if checkpoint is not None and checkpoint != '':
            checkpoint = torch_ext.load_checkpoint(checkpoint)
            self.model.load_state_dict(checkpoint['model'])
        else:
            print("No valid checkpoint has been given. Aborting execution for safety!")
            return

    def run(self):
        # create runner and set the settings
        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)['params']
        self._get_env_interface()
        self._get_network_model()
        print("Done.")
        # self.env_interface.run()


@hydra.main(config_name="real_system_config", config_path="cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg.device_id = 0
    cfg.rl_device = f'cuda:{0}'

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    rlg_trainer = SystemInterface(cfg, cfg_dict)
    rlg_trainer.run()


if __name__ == '__main__':
    parse_hydra_configs()
