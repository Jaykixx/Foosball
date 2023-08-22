import os
import copy
import torch

from environments.Foosball.base import FoosballTask

from utils.custom_runner import CustomRunner as Runner
import time


class FoosballSelfPlay(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._num_agents = 2
        self._num_actions = 8
        super().__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

        self.num_opponents = self._task_cfg["env"].get("num_opponents", 1)
        self.opponent_obs_ranges = [
            i * self._num_envs // self.num_opponents for
            i in range(self.num_opponents + 1)
        ]

        # on reset there are no observations available
        self._full_actions = self._duplicate_actions

        self.agents = None

    def add_opponent_action(self, actions):
        op_actions = tuple([
            torch.atleast_2d(
                self.agents[i].get_action(
                    {"obs": self.inv_obs_buf[
                        self.opponent_obs_ranges[i]:self.opponent_obs_ranges[i + 1],
                        ...
                    ]}
                ).detach()
            )
            for i in range(self.num_opponents)
        ])
        return torch.cat((actions, torch.cat(op_actions, 0)), 1)

    def cleanup(self) -> None:
        super().cleanup()
        self.inv_obs_buf = torch.zeros_like(self.obs_buf)

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
            (fig_pos_w, fig_pos_b, fig_vel_w, fig_vel_b, ball_pos, ball_vel), dim=-1
        )

        self.inv_obs_buf = torch.cat(
            (fig_pos_b, fig_pos_w, fig_vel_b, fig_vel_w, -ball_pos, -ball_vel), dim=-1
        ).clone()

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def _order_joints(self) -> list:
        joints = self.robot.dof_paths_W + self.robot.dof_paths_B
        active_dofs = []
        for j in joints:
            active_dofs.append(self._robots.get_dof_index(j))
        return active_dofs

    def post_reset(self):
        # first half of actions are white, second are black
        self.active_dofs = self._order_joints()
        super().post_reset()

    def reset(self):
        if self.agents is None:
            self.create_agent(self._cfg['train'])
        super().reset()

    def create_agent(self, config) -> None:
        r = Runner()
        r.load(config)
        # create opponents in eval mode
        r.params["opponent"] = True

        self.agents = [r.create_player() for _ in range(self.num_opponents)]

    def prepare_opponent(self):
        self._full_actions = self.add_opponent_action

    def full_actions(self, actions):
        return self._full_actions(actions)

    @staticmethod
    def _duplicate_actions(actions):
        return torch.cat((actions, actions), 1)

    def update_weights(self, indices, weights):
        for i in indices:
            self.agents[i%self.num_opponents].set_weights(weights)
