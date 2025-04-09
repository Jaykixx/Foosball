import torch

from environments.foosball.foosball_selfplay import FoosballSelfPlay
from utilities.models.kalman_filter import KalmanFilter


class FoosballKeeperSelfPlay(FoosballSelfPlay):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._num_agents = 2
        if not hasattr(self, "_num_actions"):
            self._num_actions = 2

        super().__init__(name, sim_config, env, offset)

    def _order_joints(self) -> list:
        joint_names = self.robot.dof_paths_W + self.robot.dof_paths_B
        joints = [name for name in joint_names if 'Keeper' in name]
        active_joint_dofs = []
        for j in joints:
            active_joint_dofs.append(self._robots.get_dof_index(j))
        return active_joint_dofs

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_resets = len(env_ids)
        device = self._device

        sign = torch.sign(torch.rand(num_resets, device=device) - 0.5)
        init_ball_pos = self._init_ball_position[env_ids].clone()
        x_offset = torch.rand_like(init_ball_pos[:, 0], device=device) * 2 - 1
        init_ball_pos[:, 0] -= sign * (0.497 + 0.005 * x_offset)
        y_offset = torch.rand_like(init_ball_pos[:, 1], device=device) * 2 - 1
        y_offset *= self.reset_position_noise
        init_ball_pos[:, 1] = sign * y_offset
        self._balls.set_world_poses(
            init_ball_pos + self._env_pos[env_ids],
            self._init_ball_rotation[env_ids].clone(),
            indices=indices
        )

        init_ball_vel = self._init_ball_velocities[env_ids].clone()
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def get_observations(self) -> dict:
        fig_pos = self._robots.get_joint_positions(joint_indices=self.active_joint_dofs, clone=False)
        fig_vel = self._robots.get_joint_velocities(joint_indices=self.active_joint_dofs, clone=False)
        fig_pos_w = fig_pos[:, :self.num_actions]
        fig_pos_b = fig_pos[:, self.num_actions:]
        fig_vel_w = fig_vel[:, :self.num_actions]
        fig_vel_b = fig_vel[:, self.num_actions:]

        # Observe game ball in x-, y-axis
        ball_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_pos[:, :2] - self._env_pos[:, :2]
        ball_vel = self._balls.get_velocities(clone=False)[:, :2]

        self.obs_buf = torch.cat(
            (fig_pos_w, fig_vel_w, fig_pos_b[:, 0:1], fig_vel_b[:, 0:1], ball_pos, ball_vel), dim=-1
        )

        self.inv_obs_buf = torch.cat(
            (fig_pos_b, fig_vel_b, fig_pos_w[:, 0:1], fig_vel_w[:, 0:1],  -ball_pos, -ball_vel), dim=-1
        ).clone()

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }

        if self.capture:
            self.capture_image()
        return observations

    def _fig_to_ball_reward(self, ball_pos):
        # Only when Ball is stationary and spawns on white side
        vel = self._balls.get_velocities(clone=False)[:, :2]
        vel = torch.norm(vel, 2, dim=-1)
        pull_fig_mask = torch.min(ball_pos[:, 0] > 0, vel < 0.1)

        fig_pos_dist = self._compute_fig_to_ball_distances(ball_pos)[0]
        fig_pos_rew = - (1 - torch.exp(-6 * fig_pos_dist))
        fig_pos_rew[~pull_fig_mask] = 0
        return fig_pos_rew
