from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch import *
from environments.foosball.foosball_base import FoosballTask
from utilities.robots.foosball import Foosball
import torch


class FoosballScoringTask(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_actions"):
            self._num_actions = 2
        if not hasattr(self, "_dof"):
            self._dof = 2
        if not hasattr(self, "_num_objects"):
            self._num_objects = 2

        super(FoosballScoringTask, self).__init__(name, sim_config, env, offset)

    def post_reset(self) -> None:
        self.active_joint_dofs = []
        self.active_joint_dofs.append(self._robots.get_dof_index("Keeper_W_PrismaticJoint"))
        self.active_joint_dofs.append(self._robots.get_dof_index("Keeper_W_RevoluteJoint"))

        super(FoosballScoringTask, self).post_reset()

    def _calculate_metrics(self):
        wins, losses, timeouts = super()._calculate_metrics()

        # pos = self._balls.get_world_poses(clone=False)[0]
        # ball_pos = pos - self._env_pos

        # # Optional Reward: Ball near opponent goal
        # self.rew_buf += self._dist_to_goal_reward(ball_pos)

        # Optional Reward: Regularization of actions
        # self.rew_buf += 0.5 * self._compute_action_regularization()

        # # Optional Reward: Pull figures to ball
        # self.rew_buf += self._fig_to_ball_reward(ball_pos)

        return wins, losses, timeouts