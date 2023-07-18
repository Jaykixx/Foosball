from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames


class CustomVecEnvRLGames(VecEnvRLGames):
    """ Wraps the default Isaac Sim RL Environment to allow for customization
    and higher flexibility, e.g. access to private properties. """

    @property
    def task(self):
        return self._task

    @property
    def progress_buffer(self):
        return self._task.progress_buf
