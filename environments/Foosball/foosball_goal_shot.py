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

    # def get_observations(self) -> dict:
    #     # Observe figurines
    #     fig_pos = self._robots.get_joint_positions(joint_indices=self.observations_dofs, clone=False)
    #
    #     # Observe game ball in x-, y-axis
    #     ball_pos = self._balls.get_world_poses(clone=False)[0]
    #     ball_pos = ball_pos[:, :2] - self._env_pos[:, :2]
    #
    #     if self.apply_kalman_filter:
    #         self.kalman.predict()
    #         ball_pos, ball_vel = self.kalman.state[:2], self.kalman.state[2:]
    #         self.kalman.correct(ball_pos)
    #     else:
    #         ball_vel = self._balls.get_velocities(clone=False)[:, :2]
    #
    #     self.obs_buf = torch.cat(
    #         (fig_pos, ball_pos, ball_vel), dim=-1
    #     )
    #
    #     observations = {
    #         self._robots.name: {
    #             "obs_buf": self.obs_buf
    #         }
    #     }
    #
    #     if self.capture:
    #         self.capture_image()
    #     return observations

    def post_reset(self) -> None:
        self.active_dofs = []
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_PrismaticJoint"))
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_RevoluteJoint"))

        super(FoosballGoalShotTask, self).post_reset()

    def _calculate_metrics(self, ball_pos) -> None:
        super()._calculate_metrics(ball_pos)

        vel = self._balls.get_velocities(clone=False)[:, :2]
        vel = torch.norm(vel, 2, dim=-1)

        # Award closeness to opponent goal
        dist_to_b_goal, _ = self._compute_ball_to_goal_distances(ball_pos)
        dist_to_goal_rew = torch.exp(-6*dist_to_b_goal)  # - torch.exp(-6*dist_to_w_goal)
        self.rew_buf += dist_to_goal_rew

        # Regularization of actions
        self.rew_buf += 0.1 * self._compute_action_regularization()

        pull_fig_mask = vel < 0.1
        if torch.sum(pull_fig_mask) > 0:
            fig_pos_dist = self._compute_fig_to_ball_distances(ball_pos)[0]
            fig_pos_rew = torch.exp(-6*fig_pos_dist)
            self.rew_buf[pull_fig_mask] += - (1 - fig_pos_rew[pull_fig_mask])
