from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch import *
from environments.Foosball.base import FoosballTask
from utils.models.dmp import DMP
from utils.robots.foosball import Foosball
import torch


class FoosballBlockingTask(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_observations"):
            self._num_observations = 8
        if not hasattr(self, "_num_actions"):
            self._num_actions = 2
        if not hasattr(self, "_dof"):
            self._dof = 2

        super(FoosballBlockingTask, self).__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        device = self.device

        init_ball_pos = self._init_ball_position[env_ids].clone()
        init_ball_pos[:, 0] -= 0.48
        y_offset = (torch.rand_like(init_ball_pos[:, 1], device=device) * 2 - 1)
        y_offset *= self.reset_position_noise
        init_ball_pos[:, 1] = y_offset
        self._balls.set_world_poses(
            init_ball_pos + self._env_pos[env_ids], indices=indices
        )

        # Reset ball velocity to vector of random magnitude aimed at goal
        total_vel = torch.rand(len(indices), device=device) \
                    * self.reset_velocity_noise \
                    + (10 - self.reset_velocity_noise)  # max 10 m/s
        d1 = y_offset.abs() - 0.205/2 + 2*self._ball_radius
        d2 = y_offset.abs() + 0.205/2 - 2*self._ball_radius
        xd1 = torch.sqrt(total_vel**2 / (1 + (d1/1.08)**2))
        xd2 = torch.sqrt(total_vel**2 / (1 + (d2/1.08)**2))
        xd_min = torch.minimum(xd1, xd2)
        xd_max = torch.maximum(xd1, xd2)
        xd = torch.rand_like(xd_min, device=device) * (xd_max - xd_min) + xd_min
        yd = - torch.sign(y_offset) * torch.sqrt(total_vel**2 - xd**2)
        z = torch.zeros_like(xd, device=device)
        init_ball_vel = torch.stack((xd, yd, z, z, z, z), dim=-1)
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def post_reset(self) -> None:
        self.active_dofs = []
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_PrismaticJoint"))
        self.active_dofs.append(self._robots.get_dof_index("Keeper_W_RevoluteJoint"))

        super(FoosballBlockingTask, self).post_reset()

    # def pre_physics_step(self, actions) -> None:
    #     reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    #     if len(reset_env_ids) > 0:
    #         self.reset_idx(reset_env_ids)
    #
    #     self.actions = actions.clone().to(self.device)
    #
    #     limit = self._robot_vel_limit[:, self.active_dofs]
    #     actions = torch.clamp(actions, -limit, limit)
    #     self._robots.set_joint_velocity_targets(
    #         actions, joint_indices=self.active_dofs
    #     )

    def calculate_metrics(self) -> None:
        pos = self._balls.get_world_poses(clone=False)[0]
        pos = pos - self._env_pos

        mask_y = torch.min(-0.08 < pos[:, 1], pos[:, 1] < 0.08)

        # Check white goal hit
        mask_x = 0.62 < pos[:, 0]
        loss_mask = torch.min(mask_x, mask_y)
        self.rew_buf[loss_mask] = -1

        # Ball still in game
        self.rew_buf[loss_mask == 0] = 0

        # Check Termination penalty
        limit = self._init_ball_position[0, 2] + self.termination_height
        mask_z = pos[:, 2] > limit
        self.rew_buf[mask_z] = - self.termination_penalty