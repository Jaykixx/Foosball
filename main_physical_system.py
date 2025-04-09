from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from rl_games.common import env_configurations, vecenv

from utilities.custom_runner import CustomRunner as Runner
from utilities.task_util import initialize_physical_task
from utilities.environment import PhysicalEnv

from omegaconf import DictConfig
import datetime
import hydra
import sys
import os


class SystemInterface:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        # `create_rlgpu_env` is an environment construction function
        # which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        self.cfg_dict["task"]["test"] = True
        self.cfg.test = True

        # register the rl-games adapter to use inside the runner
        vecenv.register('RLGPU',
                        lambda config_name,
                               num_actors,
                               **kwargs: RLGPUEnv(config_name, num_actors, **kwargs)
                            )
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: env
        })

        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self):
        # create runner and set the settings
        runner = Runner()
        runner.load(self.rlg_config_dict)
        runner.reset()

        runner.run_physical_system({
            'train': not self.cfg.test,
            'play': self.cfg.test,
            'checkpoint': self.cfg.checkpoint,
            'sigma': None
        })


@hydra.main(config_name="config_physical_system", config_path="cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    env = PhysicalEnv(cfg.rl_device)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    initialize_physical_task(cfg_dict, env)

    system_interface = SystemInterface(cfg, cfg_dict)
    system_interface.launch_rlg_hydra(env)
    system_interface.run()
    env.shutdown()


if __name__ == '__main__':
    parse_hydra_configs()
