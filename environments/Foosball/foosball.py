import os
import numpy as np
from environments.Foosball.base import FoosballTask
from utils.robots.foosball import Foosball
import torch


# TODO: Add player differentiation
# TODO: Incorporate Multiagent Setup
class FoosballGame(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        super(FoosballGame, self).__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

    def get_observations(self) -> dict:
        # Observe figurines (only white) TODO: Normalize?
        fig_pos = self._robots.get_joint_positions(clone=False)
        fig_vel = self._robots.get_joint_velocities(clone=False)

        # Observe game ball in x-, y-axis
        ball_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_pos[:, :2] - self._env_pos[:, :2]
        ball_vel = self._balls.get_velocities(clone=False)[:, :2]

        self.obs_buf[..., 0:16] = fig_pos
        self.obs_buf[..., 16:32] = fig_vel
        self.obs_buf[..., 32:34] = ball_pos
        self.obs_buf[..., 34:36] = ball_vel

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations
