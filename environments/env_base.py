from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames


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

    @property
    def task(self):
        return self._task

    @property
    def progress_buffer(self):
        return self._task.progress_buf
