from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch import *
from environments.Foosball.base import FoosballTask
from utilities.models.dmp import DMP
from utilities.robots.foosball import Foosball
import torch


class FoosballScoringTask(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_observations"):
            self._num_observations = 6
        if not hasattr(self, "_num_actions"):
            self._num_actions = 2
        if not hasattr(self, "_dof"):
            self._dof = 2

        super(FoosballScoringTask, self).__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]

    def post_reset(self) -> None:
        self.active_dofs = []
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_PrismaticJoint"))
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_RevoluteJoint"))

        super(FoosballScoringTask, self).post_reset()

    def _calculate_metrics(self, ball_pos) -> None:
        super()._calculate_metrics(ball_pos)

        # Optional Reward: Ball near opponent goal
        self.rew_buf += self._dist_to_goal_reward(ball_pos)

        # Optional Reward: Regularization of actions
        self.rew_buf += 0.1 * self._compute_action_regularization()

        # Optional Reward: Pull figures to ball
        self.rew_buf += self._fig_to_ball_reward(ball_pos)
