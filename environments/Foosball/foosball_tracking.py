from environments.Foosball.base import FoosballTask
import torch


class FoosballTrackingTask(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_observations"):
            self._num_observations = 1
        if not hasattr(self, "_num_actions"):
            self._num_actions = 1
        if not hasattr(self, "_dof"):
            self._dof = 1

        super(FoosballTrackingTask, self).__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        device = self.device
        num_resets = len(env_ids)

        init_ball_pos = self._init_ball_position[env_ids].clone()
        offset = torch.rand(num_resets, device=device)
        init_ball_pos[:, 0] = offset * 0.8 - 0.4
        init_ball_rot = self._init_ball_rotation[env_ids].clone()
        self._balls.set_world_poses(
            init_ball_pos + self._env_pos[env_ids], init_ball_rot, indices=indices
        )

        # Reset ball velocity to vector of random magnitude aimed at goal
        sign = torch.sign(torch.rand(num_resets, device=device) - 0.5)
        yd = torch.rand(num_resets, device=device)
        init_ball_vel = self._init_ball_velocities[env_ids].clone()
        init_ball_vel[..., :1] = 0
        init_ball_vel[..., 1] = sign * yd
        init_ball_vel[..., 2:] = 0
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def post_reset(self) -> None:
        self.active_dofs = []
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_PrismaticJoint"))

        super().post_reset()

    def get_observations(self) -> dict:
        # Observe game ball in x-, y-axis
        ball_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_pos[:, 1:2] - self._env_pos[:, 1:2]

        self.obs_buf = ball_pos

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }

        if self.capture:
            self.capture_image()
        return observations

    def _calculate_metrics(self, ball_pos) -> None:
        super()._calculate_metrics(ball_pos)
        self.rew_buf[:] = 0  # Ignore accidental goals in favor of tracking

        # Optional Reward: Pull figures to ball
        self.rew_buf += self._fig_to_ball_reward(ball_pos)
