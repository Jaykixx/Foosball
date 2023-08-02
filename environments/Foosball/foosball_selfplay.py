import os
import copy
import torch

from environments.Foosball.base import FoosballTask

from utils.custom_runner import CustomRunner as Runner

class FoosballSelfPlay(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._num_agents = 2
        self._num_actions = 8
        super().__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

        # on reset there are no observations available
        self._full_actions = self._duplicate_actions

        self.agent = None

    def add_opponent_action(self, actions):
        inverted_obs = self._invert_obs(self.obs_bufs)
        op_actions = torch.atleast_2d(self.agent.get_action(inverted_obs))

        return torch.cat((actions, op_actions), 1)

    @staticmethod
    def _invert_obs(obs):
        inverted_obs = copy.copy(obs)
        inverted_obs['obs'] = -inverted_obs['obs']
        return inverted_obs  # TODO: Implement

    def get_observations(self) -> dict:
        # TODO: Normalize?
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

    def _order_joints(self) -> list:
        joints = [
                'Keeper_W_PrismaticJoint',
                'Keeper_W_RevoluteJoint',
                'Defense_W_PrismaticJoint',
                'Defense_W_RevoluteJoint',
                'Mid_W_PrismaticJoint',
                'Mid_W_RevoluteJoint',
                'Offense_W_PrismaticJoint',
                'Offense_W_RevoluteJoint',
                'Keeper_B_PrismaticJoint',
                'Keeper_B_RevoluteJoint',
                'Defense_B_PrismaticJoint',
                'Defense_B_RevoluteJoint',
                'Mid_B_PrismaticJoint',
                'Mid_B_RevoluteJoint',
                'Offense_B_PrismaticJoint',
                'Offense_B_RevoluteJoint'
            ]
        active_dofs = []
        for j in joints:
            active_dofs.append(self._robots.get_dof_index(j))
        return active_dofs

    def post_reset(self):
        # first half of actions are white, second are black
        self.active_dofs = self._order_joints()
        super().post_reset()

    def reset(self):
        if self.agent is None:
            self.create_agent(self._cfg['train'])
        super().reset()

    def create_agent(self, config) -> None:
        runner = Runner()
        runner.load(config)

        self.agent = runner.create_player()
        self.agent.model.eval()

    def prepare_opponent(self):
        self._full_actions = self.add_opponent_action

    def full_actions(self, actions):
        return self._full_actions(actions)

    @staticmethod
    def _duplicate_actions(actions):
        return torch.cat((actions, actions), 1)

    def update_weights(self, indices, weights):
        self.agent.set_weights(weights)
