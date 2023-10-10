from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUEnv


class CustomVecEnvRLGames(VecEnvRLGames):
    """ Wraps the default Isaac Sim RL Environment to allow for customization
    and higher flexibility, e.g. access to private properties. """

    def __init__(
            self, headless: bool,
            sim_device: int = 0,
            enable_livestream: bool = False,
            enable_viewport: bool = False
    ) -> None:
        super(CustomVecEnvRLGames, self).__init__(
            headless, sim_device, enable_livestream, enable_viewport
        )
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.isaac")

        # enable multi agent settings
        self.full_actions = lambda actions: actions

    def set_task(self, **kwargs) -> None:
        super().set_task(**kwargs)
        if self._task.num_agents > 1:
            self.full_actions = lambda action: self._task.full_actions(action)

    @property
    def task(self):
        return self._task

    @property
    def progress_buffer(self):
        return self._task.progress_buf

    def get_number_of_agents(self):
        return self._task.num_agents

    def step(self, actions):
        return super().step(self.full_actions(actions))

    def reset(self):
        obs = super().reset()
        if self._task.num_agents > 1:
            self._task.prepare_opponent()
            self._task.obs_bufs = obs
        return obs

    def set_weights(self, indices, weights):
        self._task.update_weights(indices, weights)


class SelfPlayRLGPUEnv(RLGPUEnv):
    def set_weights(self, indices, weights) -> None:
        self.env.set_weights(indices, weights)
