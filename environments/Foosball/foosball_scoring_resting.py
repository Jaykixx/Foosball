from environments.Foosball.foosball_scoring import FoosballScoringTask
import torch


class FoosballScoringRestingTask(FoosballScoringTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_observations"):
            self._num_observations = 8

        super(FoosballScoringRestingTask, self).__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        device = self.device

        init_ball_pos = self._init_ball_position[env_ids].clone()
        x_offset = torch.rand_like(init_ball_pos[:, 0], device=device) * 2 - 1
        init_ball_pos[:, 0] = 0.497 + 0.005 * x_offset
        y_offset = torch.rand_like(init_ball_pos[:, 1], device=device) * 2 - 1
        y_offset *= self.reset_position_noise
        init_ball_pos[:, 1] = y_offset
        self._balls.set_world_poses(
            init_ball_pos + self._env_pos[env_ids],
            self._init_ball_rotation[env_ids].clone(),
            indices=indices
        )

        self._balls.set_velocities(self._init_ball_velocities[env_ids].clone(), indices=indices)
