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

    def post_reset(self) -> None:
        self.active_dofs = []
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_PrismaticJoint"))
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_RevoluteJoint"))

        super(FoosballGoalShotTask, self).post_reset()

    def _fig_to_ball_reward(self, ball_pos):
        # Only when Ball is stationary
        vel = self._balls.get_velocities(clone=False)[:, :2]
        vel = torch.norm(vel, 2, dim=-1)
        pull_fig_mask = vel < 0.1

        fig_pos_dist = self._compute_fig_to_ball_distances(ball_pos)[0]
        fig_pos_rew = - (1 - torch.exp(-6 * fig_pos_dist))
        fig_pos_rew[~pull_fig_mask] = 0
        return fig_pos_rew

    def _calculate_metrics(self, ball_pos) -> None:
        super()._calculate_metrics(ball_pos)

        # Optional Reward: Ball near opponent goal
        self.rew_buf += self._dist_to_goal_reward(ball_pos)

        # Optional Reward: Regularization of actions
        self.rew_buf += 0.1 * self._compute_action_regularization()

        # Optional Reward: Pull figures to ball
        self.rew_buf += self._fig_to_ball_reward(ball_pos)
