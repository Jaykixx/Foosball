from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch import *
from environments.foosball.foosball_scoring import FoosballScoringTask
from utilities.robots.foosball import Foosball
import torch


class FoosballScoringIncomingTask(FoosballScoringTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        FoosballScoringTask.__init__(self, name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

    def reset_ball(self, env_ids):
        if self.apply_kalman_filter:
            self.kalman.reset_idx(env_ids)
        indices = env_ids.to(dtype=torch.int32)
        device = self.device

        init_ball_pos = self._init_ball_position[env_ids].clone()
        offset = torch.rand(len(env_ids), device=self._device)
        init_ball_pos[:, 0] -= offset * 0.4
        y_offset = torch.rand_like(init_ball_pos[:, 1], device=device) * 2 - 1
        y_offset *= self.reset_position_noise
        init_ball_pos[:, 1] = y_offset
        init_ball_pos[:, 2] += 0.005
        init_ball_rot = self._init_ball_rotation[env_ids].clone()
        self._balls.set_world_poses(
            init_ball_pos + self._env_pos[env_ids], init_ball_rot, indices=indices
        )
        if self.apply_kalman_filter:
            self.kalman.state[env_ids, :2] = init_ball_pos[:, :2].unsqueeze(-1)

        # Reset ball velocity to vector of random magnitude aimed at goal
        total_vel = torch.rand(len(indices), device=device) \
                    * self.reset_velocity_noise \
                    + (7 - self.reset_velocity_noise)  # max 10 m/s
        init_ball_vel = self._init_ball_velocities[env_ids].clone()
        d1 = y_offset.abs() - 0.205 / 2 + 2 * self._ball_radius
        d2 = y_offset.abs() + 0.205 / 2 - 2 * self._ball_radius
        x_offset = 0.6 - init_ball_pos[:, 0]
        xd1 = torch.sqrt(total_vel**2 / (1 + (d1 / x_offset) ** 2))
        xd2 = torch.sqrt(total_vel**2 / (1 + (d2 / x_offset) ** 2))
        xd_min = torch.minimum(xd1, xd2)
        xd_max = torch.maximum(xd1, xd2)
        xd = torch.rand_like(xd_min, device=self._device) * (xd_max - xd_min) + xd_min
        yd = - torch.sign(y_offset) * torch.sqrt(total_vel**2 - xd ** 2)
        init_ball_vel[..., 0] = xd
        init_ball_vel[..., 1] = yd
        init_ball_vel[..., 2:] = 0
        self._balls.set_velocities(init_ball_vel, indices=indices)
