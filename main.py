from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path

from environments.env_base import CustomVecEnvRLGames, SelfPlayRLGPUEnv
from utilities.custom_runner import CustomRunner as Runner
from rl_games.common import env_configurations, vecenv
from utilities.task_util import initialize_task

from omegaconf import DictConfig
import datetime
import hydra
import os


class RLGTrainer:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        # `create_rlgpu_env` is an environment construction function
        # which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        self.cfg_dict["task"]["test"] = self.cfg.test

        # register the rl-games adapter to use inside the runner
        if "SelfPlay" in self.cfg_dict["task_name"]:
            vecenv.register('RLGPU',
                            lambda config_name,
                            num_actors,
                            **kwargs: SelfPlayRLGPUEnv(config_name, num_actors, **kwargs)
            )
        else:
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
        runner = Runner(RLGPUAlgoObserver())
        runner.load(self.rlg_config_dict)
        runner.reset()

        # dump config dict
        experiment_dir = os.path.join('runs', self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        runner.run({
            'train': not self.cfg.test,
            'play': self.cfg.test,
            'checkpoint': self.cfg.checkpoint,
            'sigma': None
        })


@hydra.main(config_name="config", config_path="cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f'cuda:{rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = CustomVecEnvRLGames(headless=headless,
                              sim_device=cfg.device_id,
                              enable_livestream=cfg.enable_livestream,
                              enable_viewport=enable_viewport
    )

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    task = initialize_task(cfg_dict, env)

    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()


if __name__ == '__main__':
    parse_hydra_configs()
