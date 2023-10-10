import os
import copy
import torch

from environments.Foosball.base import FoosballTask

from utilities.custom_runner import CustomRunner as Runner
import time


class FoosballSelfPlay(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._num_agents = 2
        if not hasattr(self, "_num_actions"):
            self._num_actions = 8
        if not hasattr(self, "_dof"):
            self._dof = 2 * self._num_actions
        if not hasattr(self, "_num_observations"):
            self._num_observations = 2 * self._dof + 4

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

        if self.capture:
            self.capture_image()
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
        if config['params']['load_checkpoint']:
            for agent in self.agents:
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
            self.agents[i%self.num_opponents].set_weights(weights)

    def _calculate_metrics(self, ball_pos) -> None:
        super()._calculate_metrics(ball_pos)

        # # Reward closeness to opponent goal
        # dist_to_b_goal, _ = self._compute_ball_to_goal_distances(ball_pos)
        # dist_to_goal_rew = torch.exp(-6*dist_to_b_goal)  # - torch.exp(-6*dist_to_w_goal)
        # self.rew_buf += dist_to_goal_rew
        #
        # # Regularization of actions
        # self.rew_buf += self._compute_action_regularization()
        #
        # # Pull figures to ball
        # fig_pos_dist = self._compute_fig_to_ball_distances(ball_pos)
        # fig_pos_dist = torch.stack(fig_pos_dist)
        # fig_pos_rew = torch.exp(-6*fig_pos_dist).mean(dim=0)
        # self.rew_buf += - (1 - fig_pos_rew)
        #
        # dofs = ["Keeper_W_RevoluteJoint", "Defense_W_RevoluteJoint", "Mid_W_RevoluteJoint", "Offense_W_RevoluteJoint"]
        # dof_ids = [self._robots.get_dof_index(dof) for dof in dofs]
        # fig_rot = self._robots.get_joint_positions(joint_indices=dof_ids, clone=False)
        # self.rew_buf += 0.1 * torch.mean(torch.cos(fig_rot) - 1, dim=-1)
