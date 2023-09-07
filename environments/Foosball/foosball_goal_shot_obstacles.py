from environments.Foosball.foosball_goal_shot import FoosballGoalShotTask
import torch


class FoosballGoalShotObstacleTask(FoosballGoalShotTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_observations"):
            self._num_observations = 9
        if not hasattr(self, "_num_actions"):
            self._num_actions = 2
        if not hasattr(self, "_dof"):
            self._dof = 2

        super(FoosballGoalShotObstacleTask, self).__init__(name, sim_config, env, offset)

    def post_reset(self) -> None:
        self.observations_dofs = [self._robots.get_dof_index("Defense_B_PrismaticJoint")]
        super(FoosballGoalShotObstacleTask, self).post_reset()

    def reset_idx(self, env_ids):
        id_def = self._robots.get_dof_index("Defense_B_PrismaticJoint")
        value_range = self.dof_range[env_ids, id_def]

        value = torch.rand(len(env_ids), device=self.device) * value_range
        value += self.robot_lower_limit[id_def] + self.dof_offset[env_ids, id_def]

        self._default_joint_pos[env_ids, id_def] = value
        super().reset_idx(env_ids)

