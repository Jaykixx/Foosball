from omni.isaac.core.prims import RigidPrimView, XFormPrimView

from environments.foosball.foosball_selfplay import FoosballSelfPlay

import numpy as np
import torch


class FoosballMixedSelfPlay(FoosballSelfPlay):

    def __init__(self, name, sim_config, env, offset):
        FoosballSelfPlay.__init__(self, name, sim_config, env, offset)

        self.active_dof_mask = torch.zeros((self._num_envs, self._dof), dtype=torch.bool, device=self.device)

    def get_obj_centric_observations(self):
        obj_obs = []
        inv_obj_obs = []
        for name, value in self.active_rods.items():
            # TODO: Rescale to table size
            sign = 1 if 'W' in name else -1  # Joints for black are mirrored so signs are needed

            fig_tpos = self.robot.figure_positions[name][None].repeat_interleave(self.num_envs, 0)
            fig_tpos[:, 1] += sign * self._robots.get_joint_positions(joint_indices=[value['pris_id']], clone=False)

            dof_rpos = sign * self._robots.get_joint_positions(joint_indices=[value['rev_id']], clone=False)
            fig_rpos = dof_rpos[..., None].repeat_interleave(fig_tpos.shape[-1], -1)

            fig_tvel = torch.zeros_like(fig_tpos)
            fig_tvel[:, 1] = sign * self._robots.get_joint_velocities(joint_indices=[value['pris_id']], clone=False)

            dof_rvel = sign * self._robots.get_joint_velocities(joint_indices=[value['rev_id']], clone=False)
            fig_rvel = dof_rvel[..., None].repeat_interleave(fig_tvel.shape[-1], -1)

            one_hot_encoding = torch.zeros((self.num_envs, self._num_obj_types, fig_tpos.shape[-1]), device=self.device)
            inv_one_hot_encoding = torch.zeros_like(one_hot_encoding)
            if 'W' in name:
                rod_idx = self.robot.rod_paths_W.index('White/' + name)
                mask = self.white_rods_mask[:, rod_idx]
                one_hot_encoding[mask, 0] = 1
                inv_one_hot_encoding[mask, 1] = 1
            elif 'B' in name:
                rod_idx = self.robot.rod_paths_B.index('Black/' + name)
                mask = self.black_rods_mask[:, rod_idx]
                one_hot_encoding[mask, 1] = 1
                inv_one_hot_encoding[mask, 0] = 1

            fig_obs = torch.cat((
                one_hot_encoding, fig_tpos, fig_rpos, fig_tvel, fig_rvel,
            ), dim=1).transpose(1, 2)

            # TODO: Inverse Order ???
            inv_fig_obs = torch.cat((
                inv_one_hot_encoding, -fig_tpos, -fig_rpos, -fig_tvel, -fig_rvel,
            ), dim=1).transpose(1, 2)

            obj_obs.append(fig_obs)
            inv_obj_obs.append(inv_fig_obs)

        ball_obs = torch.zeros((self.num_envs, self._num_obj_features + self._num_obj_types), device=self.device)
        ball_pos, ball_vel = self.get_ball_observation()
        ball_obs[..., self._num_obj_types-1] = 1
        ball_obs[..., self._num_obj_types:self._num_obj_types+2] = ball_pos
        ball_obs[..., -3:-1] = ball_vel
        obj_obs.append(ball_obs[:, None])
        inv_obj_obs.append(-ball_obs[:, None])

        self.obs_buf = torch.cat(obj_obs, dim=1)
        self.inv_obs_buf = torch.cat(inv_obj_obs, dim=1)

    def hide_rods(self):
        for idx in range(self.num_envs):
            flags = torch.zeros(1, device=self.device, dtype=torch.bool)
            for rod_name in self.hidden_rods[idx]:
                rod_prim = XFormPrimView(
                    prim_paths_expr=f"/World/envs/env_{idx}/Foosball/" + rod_name,
                    name="rod_view",
                    reset_xform_properties=False
                )
                rod_prim.set_visibilities(flags)

    def randomize_active_rods(self):
        white_rods = torch.rand((self._num_envs, 4), device=self.device) > 0.5
        black_rods = torch.rand((self._num_envs, 4), device=self.device) > 0.5

        # Check if at least 1 white rod is active, if not activate a random one
        num_active_rods = white_rods.sum(dim=-1)
        mask = num_active_rods == 0
        white_rods[mask, torch.randint(0, 4, (int(mask.sum()),))] = 1

        # Save to filter object centric one hot encoding
        self.white_rods_mask = white_rods
        self.black_rods_mask = black_rods

        rod_paths_W = np.asarray(self.robot.rod_paths_W)
        rod_paths_B = np.asarray(self.robot.rod_paths_B)
        self.hidden_rods = []
        for n in range(self.num_envs):
            wr = list(rod_paths_W[white_rods[n] == 1])
            br = list(rod_paths_B[black_rods[n] == 1])
            self.hidden_rods.append(wr + br)
        self.hide_rods()

        # Duplicate to match prismatic and revolute joints
        white_dofs = torch.cat((white_rods, white_rods), dim=-1)
        black_dofs = torch.cat((black_rods, black_rods), dim=-1)

        # Required to override actions of inactive joints
        self.active_dof_mask = torch.cat((white_dofs, black_dofs), dim=-1)

    def post_reset(self):
        FoosballSelfPlay.post_reset(self)

        # Move all hidden rods into horizontal position
        default_rev_pos = self._default_joint_pos[:, 8:]
        default_rev_pos[~self.active_dof_mask[:, 8:]] += np.pi/2
        self._default_joint_pos[:, 8:] = default_rev_pos


    def reset_ball(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.actions = actions.clone().to(self.device)

        if self.target_mode == 'velocity':
            self.actions[~self.active_dof_mask] = 0
            vel_limits = (self.joint_vel_limits * self.joint_vel_scale)[:, self.active_joint_dofs]
            joint_targets = torch.clamp(actions * vel_limits, -vel_limits, vel_limits)
        else:
            # To maintain equal dimension across all variants we need to override default
            # pos for all truly active joints. 'active_joint_dofs' contains all joints
            # including those of hidden rods that should stay still
            joint_range = self.joint_range[:, self.active_joint_dofs]
            joint_offset =  self.joint_offset[:, self.active_joint_dofs]
            targets = torch.clamp(
                actions * joint_range / 2 + joint_offset,
                self.joint_limits[:, self.active_joint_dofs, 0],
                self.joint_limits[:, self.active_joint_dofs, 1]
            )
            joint_targets = self._default_joint_pos.clone()
            joint_targets[self.active_dof_mask] = targets[self.active_dof_mask]

        joint_targets = self.get_delayed_joint_actions(joint_targets)
        self.joint_targets = joint_targets.clone()

        if self.low_level_controller is not None:
            self.low_level_controller.set_target(joint_targets)
        else:
            self.apply_control_target(joint_targets)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
