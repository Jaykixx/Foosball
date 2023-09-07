import os
import copy
import torch

from environments.Foosball.foosball_selfplay import FoosballSelfPlay

from utilities.custom_runner import CustomRunner as Runner
import time


class FoosballKeeperSelfPlay(FoosballSelfPlay):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._num_agents = 2
        self._num_actions = 2
        self._dof = 2 * self._num_actions
        self._num_observations = 3 * self._num_actions + 4
        super().__init__(name, sim_config, env, offset)

    def _order_joints(self) -> list:
        joint_names = self.robot.dof_paths_W + self.robot.dof_paths_B
        joints = [name for name in joint_names if 'Keeper' in name]
        active_dofs = []
        for j in joints:
            active_dofs.append(self._robots.get_dof_index(j))
        return active_dofs

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

        # # Choose random batch to start with a moving ball
        # if env_ids.size(0) > 1:
        #     perm = torch.randperm(env_ids.size(0))
        #     idx = perm[:int(len(env_ids)/2)]
        # else:
        #     if torch.rand((1,), device=device).item() - 0.5 > 0:
        #         idx = []  # [0]
        #     else:
        #         idx = []
        idx = []

        init_ball_vel = self._init_ball_velocities[env_ids].clone()

        # Reset ball velocity to vector of random magnitude aimed at goal
        if len(idx) > 0:
            total_vel = torch.rand(len(idx), device=device) \
                        * self.reset_velocity_noise \
                        + (10 - self.reset_velocity_noise)  # max 10 m/s
            d1 = y_offset[idx].abs() - 0.205 / 2 + 2 * self._ball_radius
            d2 = y_offset[idx].abs() + 0.205 / 2 - 2 * self._ball_radius
            xd1 = torch.sqrt(total_vel**2 / (1 + (d1 / 1.08) ** 2))
            xd2 = torch.sqrt(total_vel**2 / (1 + (d2 / 1.08) ** 2))
            xd_min = torch.minimum(xd1, xd2)
            xd_max = torch.maximum(xd1, xd2)
            xd = torch.rand_like(xd_min, device=device) * (xd_max - xd_min) + xd_min
            yd = - torch.sign(y_offset[idx]) * torch.sqrt(total_vel**2 - xd ** 2)
            init_ball_vel[idx, 0] = sign[idx] * xd
            init_ball_vel[idx, 1] = sign[idx] * yd
        init_ball_vel[..., 2:] = 0
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def get_observations(self) -> dict:
        # TODO: Normalize?
        fig_pos = self._robots.get_joint_positions(joint_indices=self.active_dofs, clone=False)
        fig_vel = self._robots.get_joint_velocities(joint_indices=self.active_dofs, clone=False)
        fig_pos_w = fig_pos[:, :self.num_actions]
        fig_pos_b = fig_pos[:, self.num_actions:]
        fig_vel_w = fig_vel[:, :self.num_actions]
        fig_vel_b = fig_vel[:, self.num_actions:]

        # Observe game ball in x-, y-axis
        ball_w_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_w_pos[:, :2] - self._env_pos[:, :2]
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

    def _calculate_metrics(self, ball_pos) -> None:
        super()._calculate_metrics(ball_pos)

        vel = self._balls.get_velocities(clone=False)[:, :2]
        vel = torch.norm(vel, 2, dim=-1)

        # Award closeness to opponent goal
        dist_to_b_goal, _ = self._compute_ball_to_goal_distances(ball_pos)
        dist_to_goal_rew = torch.exp(-6*dist_to_b_goal)  # - torch.exp(-6*dist_to_w_goal)
        self.rew_buf += dist_to_goal_rew

        # Regularization of actions
        self.rew_buf += self._compute_action_regularization()

        dof = [self._robots.get_dof_index("Keeper_W_PrismaticJoint")]
        fig_pos = self._robots.get_joint_positions(joint_indices=dof, clone=False)
        pull_fig_mask = torch.min(ball_pos[:, 0] > 0, vel < 0.1)
        fig_pos_dist = torch.abs(fig_pos - ball_pos[:, 1:2])
        fig_pos_rew = torch.exp(-fig_pos_dist / 0.08)
        self.rew_buf[pull_fig_mask] = - (1 - fig_pos_rew[pull_fig_mask])
