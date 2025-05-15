import os
import copy
import torch

from environments.foosball.foosball_base import FoosballTask

from utilities.custom_runner import CustomRunner as Runner
import time


class FoosballSelfPlay(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        # self._num_agents = 2
        if not hasattr(self, "_num_actions"):
            # Defines action space for AI
            self._num_actions = 8
        if not hasattr(self, "_dof"):
            # Defines action space for task - Only different for selfplay
            self._dof = 2 * self._num_actions
        if not hasattr(self, "_num_task_observations"):
            # Ball + Opponents (pos + vel)
            # self._num_task_observations = 2 * self.num_actions + 4

            # Ball + Opponents Prismatic Joints (pos + vel)
            self._num_task_observations = self.num_actions + 4

        super().__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

        self.num_opponents = self._task_cfg["env"].get("num_opponents", 1)
        self.opponents_obs_ranges = [
            i * self._num_envs // self.num_opponents for
            i in range(self.num_opponents + 1)
        ]

        # on reset there are no observations available
        self._full_actions = self._duplicate_actions

        self.opponents = None

    def add_opponent_action(self, actions):
        op_actions = tuple([
            torch.atleast_2d(
                self.opponents[i].get_action(
                    self.inv_obs_buf[
                        self.opponents_obs_ranges[i]:self.opponents_obs_ranges[i + 1],
                        ...
                    ]
                ).detach()
            )
            for i in range(self.num_opponents)
        ])
        return torch.cat((actions, torch.cat(op_actions, 0)), 1)

    def cleanup(self) -> None:
        super().cleanup()
        self.inv_obs_buf = torch.zeros_like(self.obs_buf)

    def get_observations(self) -> dict:
        dof_pos = self._robots.get_joint_positions(joint_indices=self.active_joint_dofs, clone=False)
        dof_vel = self._robots.get_joint_velocities(joint_indices=self.active_joint_dofs, clone=False)
        dof_pos_w = dof_pos[:, :self.num_actions]
        dof_pos_b = dof_pos[:, self.num_actions:]
        dof_vel_w = dof_vel[:, :self.num_actions]
        dof_vel_b = dof_vel[:, self.num_actions:]

        # Observe game ball in x-, y-axis
        ball_w_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_w_pos[:, :2] - self._env_pos[:, :2]
        ball_vel = self._balls.get_velocities(clone=False)[:, :2]

        self.obs_buf = torch.cat(
            (dof_pos_w, dof_vel_w, dof_pos_b, dof_vel_b, ball_pos, ball_vel), dim=-1
        )

        self.inv_obs_buf = torch.cat(
            (dof_pos_b, dof_vel_b, dof_pos_w, dof_vel_w, -ball_pos, -ball_vel), dim=-1
        ).clone()

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }

        if self.capture:
            self.capture_image()
        return observations

    def _order_joints(self) -> list:
        joints = self.robot.dof_paths_W + self.robot.dof_paths_B
        active_joint_dofs = []
        for j in joints:
            active_joint_dofs.append(self._robots.get_dof_index(j))
        return active_joint_dofs

    def post_reset(self):
        # first half of actions are white, second are black
        self.active_joint_dofs = self._order_joints()
        super().post_reset()

    def reset(self):
        if self.opponents is None:
            self.create_opponent(self._cfg['train'])
        super().reset()

    def create_opponent(self, config) -> None:
        r = Runner()
        r.load(config)
        # create opponents in eval mode
        r.params["opponent"] = True

        self.opponents = [r.create_player() for _ in range(self.num_opponents)]
        if config['params']['load_checkpoint']:
            for agent in self.opponents:
                agent.restore(config['params']['load_path'])

    def prepare_opponent(self):
        self._full_actions = self.add_opponent_action

    def full_actions(self, actions):
        return self._full_actions(actions)

    @staticmethod
    def _duplicate_actions(actions):
        return torch.cat((actions, actions), 1)

    def update_weights(self, indices, weights):
        for i in indices:
            self.opponents[i%self.num_opponents].set_weights(weights)

    def _calculate_metrics(self):
        wins, losses, timeouts = super()._calculate_metrics()

        pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = pos - self._env_pos

        # Optional Reward: Ball near opponent goal
        self.rew_buf += 50 * self._dist_to_goal_reward(ball_pos)

        # Optional Reward: Regularization of actions
        # self.rew_buf += 0.1 * self._compute_action_regularization()

        # Optional Reward: Pull figures to ball
        # self.rew_buf += self._fig_to_ball_reward(ball_pos)

        return wins, losses, timeouts