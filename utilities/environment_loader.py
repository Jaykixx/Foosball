from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *

from utilities.task_util import initialize_task, initialize_physical_task
from utilities.environment.env_base import CustomVecEnvRLGames
from utilities.environment import PhysicalEnv

from omegaconf import DictConfig
import datetime
import hydra
import time
import os


def _load_phys_env(cfg: DictConfig):
    global phys_env
    global phys_task

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    phys_env = PhysicalEnv(cfg.rl_device)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        project_path = os.path.join(os.path.dirname(__file__), '../')
        cfg.checkpoint = os.path.join(project_path, cfg.checkpoint)
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    phys_task = initialize_physical_task(cfg_dict, phys_env)


def _load_sim_env(cfg: DictConfig):
    global sim_env
    global sim_task

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f'cuda:{rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    sim_env = CustomVecEnvRLGames(headless=headless,
                                  sim_device=cfg.device_id,
                                  enable_livestream=cfg.enable_livestream,
                                  enable_viewport=enable_viewport
                                  )

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        project_path = os.path.join(os.path.dirname(__file__), '..')
        cfg.checkpoint = os.path.join(project_path, cfg.checkpoint)
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    sim_task = initialize_task(cfg_dict, sim_env)

    return sim_env, sim_task


def get_sim_env():
    cfg_path = os.path.join(os.path.dirname(__file__), '../cfg')
    hydra_decorator = hydra.main(cfg_path, 'config')

    global sim_env
    global sim_task
    loader = hydra_decorator(_load_sim_env)
    loader()
    return sim_env, sim_task


def get_phys_env():
    cfg_path = os.path.join(os.path.dirname(__file__), '../cfg')
    hydra_decorator = hydra.main(cfg_path, 'config_physical_system')

    global phys_env
    global phys_task
    loader = hydra_decorator(_load_phys_env)
    loader()
    return phys_env, phys_task


if __name__ == '__main__':
    sim_env, sim_task = get_sim_env()
    phys_env, phys_task = get_phys_env()

    print("Done.")
