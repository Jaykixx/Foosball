from environments.Foosball.base import FoosballTask
import torch


class FoosballGoalShotTask(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_observations"):
            self._num_observations = 8
        if not hasattr(self, "_num_actions"):
            self._num_actions = 2
        if not hasattr(self, "_dof"):
            self._dof = 2

        super(FoosballGoalShotTask, self).__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        device = self.device

        init_ball_pos = self._init_ball_position[env_ids].clone()
        init_ball_pos[:, 0] += 0.33
        y_offset = (torch.rand_like(init_ball_pos[:, 1], device=device) * 2 - 1)
        y_offset *= self.reset_position_noise
        init_ball_pos[:, 1] = y_offset
        self._balls.set_world_poses(
            init_ball_pos + self._env_pos[env_ids], indices=indices
        )

        init_ball_vel = torch.zeros((len(indices), 6), device=device)
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def post_reset(self) -> None:
        self.active_dofs = []
        self.active_dofs.append(self._robots.get_dof_index("Defense_W_PrismaticJoint"))
        self.active_dofs.append(self._robots.get_dof_index("Defense_W_RevoluteJoint"))

        super(FoosballGoalShotTask, self).post_reset()

    # def pre_physics_step(self, actions) -> None:
    #     reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    #     if len(reset_env_ids) > 0:
    #         self.reset_idx(reset_env_ids)
    #
    #     self.actions = actions.clone().to(self.device)
    #
    #     # ll = -self._robot_vel_limit[:, self.active_dofs]  # self.robot_lower_limit[self.active_dofs]
    #     # ul = self._robot_vel_limit[:, self.active_dofs]  # self.robot_upper_limit[self.active_dofs]
    #     # actions = torch.clamp(actions, ll, ul)
    #     self._robots.set_joint_velocity_targets(
    #         actions * self._robot_vel_limit[:, self.active_dofs], joint_indices=self.active_dofs
    #     )

    def calculate_metrics(self) -> None:
        pos = self._balls.get_world_poses(clone=False)[0]
        vel = self._balls.get_velocities(clone=False)[:, :2]
        pos = pos - self._env_pos
        vel = torch.norm(vel, 2, dim=-1)

        dof = [self._robots.get_dof_index("Defense_W_PrismaticJoint")]
        fig_pos = self._robots.get_joint_positions(joint_indices=dof, clone=False)
        fig_pos = - fig_pos.repeat(1, 2)
        fig_pos[:, 0] -= 0.1235
        fig_pos[:, 1] += 0.1235

        # Pull ball towards goal if ball velocity > 0
        sq_ball_pos_dist = torch.pow(pos[:, 0] + 0.62, 2) + torch.pow(pos[:, 1], 2)
        vel_reward = torch.exp(vel / 2) - 1
        reward = vel_reward * torch.exp(-torch.sqrt(sq_ball_pos_dist) / 0.08)
        reward = torch.where(vel > 2, reward + vel_reward, reward)
        reward = torch.where(vel > 4, reward + vel_reward, reward)
        reward = torch.where(vel > 6, reward + vel_reward, reward)

        # Pull nearest figure to ball if ball is not moving
        fig_pos_dist = torch.abs(fig_pos - pos[:, 1:2])
        fig_pos_dist = torch.min(fig_pos_dist, dim=-1)[0]
        fig_pos_rew = torch.exp(-fig_pos_dist / 0.08)
        reward = torch.where(vel < 0.1, reward + fig_pos_rew, reward + 1)

        mask_y = torch.min(-0.09 < pos[:, 1], pos[:, 1] < 0.09)

        # Regularization of actions
        action_penalty = torch.sum(self.actions ** 2, dim=-1)
        reward -= 0.1 * action_penalty

        # Check white goal hit
        mask_x = 0.6 < pos[:, 0]
        loss_mask = torch.min(mask_x, mask_y)
        reward[loss_mask] = 0

        # Check black goal hit
        mask_x = pos[:, 0] < -0.6
        win_mask = torch.min(mask_x, mask_y)
        time_left = self._max_episode_length - self.progress_buf
        reward[win_mask] += 1.5 * reward[win_mask] * time_left[win_mask]

        # Check Termination penalty
        limit = self._init_ball_position[0, 2] + self.termination_height
        mask_z = pos[:, 2] > limit
        reward[mask_z] -= self.termination_penalty

        termination = torch.max(loss_mask, mask_z)
        termination = torch.max(termination, win_mask)
        length_mask = self.progress_buf >= self._max_episode_length
        # reward[length_mask] -= self.termination_penalty
        self.reset_buf = torch.max(termination, length_mask)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        pass  # INCLUDED IN CALCULATE METRICS!!!
