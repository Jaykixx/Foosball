from environments.foosball.foosball_scoring_resting import FoosballScoringRestingTask
import torch


class FoosballScoringRestingObstacleTask(FoosballScoringRestingTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_task_observations"):
            self._num_observations = 7

        super(FoosballScoringRestingObstacleTask, self).__init__(name, sim_config, env, offset)

    def get_observations(self) -> dict:
        # Observe figurines
        fig_pos = self._robots.get_joint_positions(joint_indices=self.active_joint_dofs, clone=False)
        fig_vel = self._robots.get_joint_velocities(joint_indices=self.active_joint_dofs, clone=False)
        obstacles = self._robots.get_joint_positions(joint_indices=self.passive_joint_dofs, clone=False)

        # Observe game ball in x-, y-axis
        ball_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_pos[:, :2] - self._env_pos[:, :2]

        if self.apply_kalman_filter:
            self.kalman.predict()
            ball_pos, ball_vel = self.kalman.state[:2], self.kalman.state[2:]
            self.kalman.correct(ball_pos)
        else:
            ball_vel = self._balls.get_velocities(clone=False)[:, :2]

        self.obs_buf = torch.cat(
            (fig_pos, fig_vel, obstacles, ball_pos, ball_vel), dim=-1
        )

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }

        if self.capture:
            self.capture_image()
        return observations

    def post_reset(self) -> None:
        self.passive_joint_dofs = [
            self._robots.get_dof_index("Keeper_B_PrismaticJoint"),
            self._robots.get_dof_index("Defense_B_PrismaticJoint"),
            self._robots.get_dof_index("Offense_B_PrismaticJoint"),
        ]

        self.observations_dofs = self.passive_joint_dofs.copy()
        super(FoosballScoringRestingObstacleTask, self).post_reset()

    def reset_idx(self, env_ids):
        for id_def in self.passive_joint_dofs:
            value_range = self.dof_range[env_ids, id_def]

            value = torch.rand(len(env_ids), device=self.device) * value_range
            value += self.robot_lower_limit[id_def] + self.dof_offset[env_ids, id_def]

            self._default_joint_pos[env_ids, id_def] = value

        super().reset_idx(env_ids)
