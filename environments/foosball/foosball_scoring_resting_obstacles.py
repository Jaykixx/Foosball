from environments.foosball.foosball_scoring_resting import FoosballScoringRestingTask
import torch


class FoosballScoringRestingObstacleTask(FoosballScoringRestingTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_task_observations"):
            self._num_observations = 7
        if not hasattr(self, "_num_objects"):
            self._num_objects = 5

        super(FoosballScoringRestingObstacleTask, self).__init__(name, sim_config, env, offset)

    def post_reset(self) -> None:
        self.passive_joint_dofs = [
            self._robots.get_dof_index("Keeper_B_PrismaticJoint"),
            self._robots.get_dof_index("Defense_B_PrismaticJoint"),
            self._robots.get_dof_index("Offense_B_PrismaticJoint"),
        ]

        self.observed_dofs = self.passive_joint_dofs.copy()
        super(FoosballScoringRestingObstacleTask, self).post_reset()

    def reset_idx(self, env_ids):
        for id_def in self.passive_joint_dofs:
            value_range = self.joint_range[:, id_def]

            value = torch.rand(len(env_ids), device=self.device) * value_range
            value += self.joint_limits[:, id_def, 0] + self.joint_offset[:, id_def]

            self._default_joint_pos[env_ids, id_def] = value

        super().reset_idx(env_ids)
