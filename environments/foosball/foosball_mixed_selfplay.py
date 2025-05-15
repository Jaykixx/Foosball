from omni.isaac.core.prims import RigidPrimView, XFormPrimView

from environments.foosball.foosball_selfplay import FoosballSelfPlay
from utilities.functions.sampling import randint

import numpy as np
import torch


class FoosballMixedSelfPlay(FoosballSelfPlay):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        FoosballSelfPlay.__init__(self, name, sim_config, env, offset)

        self.active_dof_mask = torch.zeros((self._num_envs, self._dof), dtype=torch.bool, device=self.device)

    def get_obj_centric_observations(self):
        obj_obs = []
        inv_obj_obs = []  # Contains inverted obs for opponent query
        for name, value in self.active_rods.items():
            # TODO: Rescale to table size
            sign = -1 if 'W' in name else 1  # Joints for black are mirrored so signs are needed

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
                inv_one_hot_encoding[mask, 1] = 1  # Register as white for opponent
            elif 'B' in name:
                rod_idx = self.robot.rod_paths_B.index('Black/' + name)
                mask = self.black_rods_mask[:, rod_idx]
                one_hot_encoding[mask, 1] = 1
                inv_one_hot_encoding[mask, 0] = 1  # Register as white for opponent

            fig_obs = torch.cat((
                one_hot_encoding, fig_tpos, fig_rpos, fig_tvel, fig_rvel,
            ), dim=1).transpose(1, 2)

            # Order of objects irrelevant for object centric transformers
            #   so instead we keep object order and only switch perspective
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
        inv_ball_obs = ball_obs.clone()
        inv_ball_obs[..., self._num_obj_types:] *= -1
        obj_obs.append(ball_obs[:, None])
        inv_obj_obs.append(inv_ball_obs[:, None])

        obs = torch.cat(obj_obs, dim=1)
        inv_obs = torch.cat(inv_obj_obs, dim=1)

        # Center obs around ball
        obs[:, :-1, self._num_obj_types:self._num_obj_types+2] -= ball_pos[:, None]
        inv_obs[:, :-1, self._num_obj_types:self._num_obj_types+2] += ball_pos[:, None]
        # velocities toward ball should be positive and vice versa
        obs[:, :-1, -3:-1] *= - torch.sign(obs[:, :-1, self._num_obj_types:self._num_obj_types+2])
        inv_obs[:, :-1, -3:-1] *= - torch.sign(inv_obs[:, :-1, self._num_obj_types:self._num_obj_types + 2])

        self.obs_buf = obs.flatten(start_dim=1)
        self.inv_obs_buf = inv_obs.flatten(start_dim=1)

    def get_observations(self) -> dict:
        if self.object_centric_obs:
            self.get_obj_centric_observations()
        else:
            FoosballSelfPlay.get_observations(self)

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf,
            }
        }

        if self.capture:
            self.capture_image()
        return observations

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
        white_rods = torch.rand((self._num_envs, 4)) > 0.5
        black_rods = torch.rand((self._num_envs, 4)) > 0.5

        # Env 0 will always be 4vs4 due to clone dependency during hiding rods
        white_rods[0] = True
        black_rods[0] = True

        # Check if at least 1 white rod is active, if not activate a random one
        num_active_rods = white_rods.sum(dim=-1)
        mask = num_active_rods == 0
        white_rods[mask, torch.randint(0, 4, (int(mask.sum()),))] = 1

        # Make sure there is at least 1 defender between closest attacker and goal
        #   White
        mask = black_rods[:, -1] == True  # Opponent has offense
        white_rods[mask, torch.randint(0, 2, (int(mask.sum()),))] = 1

        mask = black_rods[:, -2] == True  # Opponent has midfield
        white_rods[mask, torch.randint(0, 3, (int(mask.sum()),))] = 1

        #   Black
        # If at least one black rod is already active, otherwise its shooting practice
        black_mask = black_rods.sum(dim=-1) >= 1
        white_mask = white_rods[:, -1] == True  # Opponent has offense
        mask = black_mask & white_mask
        black_rods[mask, torch.randint(0, 2, (int(mask.sum()),))] = 1

        white_mask = white_rods[:, -2] == True  # Opponent has midfield
        mask = black_mask & white_mask
        black_rods[mask, torch.randint(0, 3, (int(mask.sum()),))] = 1

        # Save to filter object centric one hot encoding
        self.white_rods_mask = white_rods.clone().to(self.device)
        self.black_rods_mask = black_rods.clone().to(self.device)

        rod_paths_W = np.asarray(self.robot.rod_paths_W)
        rod_paths_B = np.asarray(self.robot.rod_paths_B)
        self.hidden_rods = []
        for n in range(self.num_envs):
            wr = list(rod_paths_W[white_rods[n] == False])
            br = list(rod_paths_B[black_rods[n] == False])
            self.hidden_rods.append(wr + br)

        # Duplicate to match prismatic and revolute joints
        white_dofs = torch.cat((white_rods, white_rods), dim=-1).to(self.device)
        black_dofs = torch.cat((black_rods, black_rods), dim=-1).to(self.device)

        # Required to override actions of inactive joints
        self.active_dof_mask = torch.cat((white_dofs, black_dofs), dim=-1)

    def post_reset(self):
        self.randomize_active_rods()
        FoosballSelfPlay.post_reset(self)

        self.hide_rods()

        # Move all hidden rods into horizontal position
        default_rev_pos = self._default_joint_pos[:, 8:12]
        default_rev_pos[~self.active_dof_mask[:, 4:8]] += np.pi/2
        self._default_joint_pos[:, 8:12] = default_rev_pos

        default_rev_pos = self._default_joint_pos[:, 12:16]
        default_rev_pos[~self.active_dof_mask[:, 12:16]] += np.pi/2
        self._default_joint_pos[:, 12:16] = default_rev_pos

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_resets = len(env_ids)

        # Gather figure obs
        obj_obs = []
        for name, value in self.active_rods.items():
            sign = -1 if 'W' in name else 1  # Joints for black are mirrored so signs are needed

            fig_tpos = self.robot.figure_positions[name][None].repeat_interleave(self.num_envs, 0)
            fig_tpos[:, 1] += sign * self._robots.get_joint_positions(joint_indices=[value['pris_id']], clone=False)

            one_hot_encoding = torch.zeros((self.num_envs, self._num_obj_types, fig_tpos.shape[-1]), device=self.device)
            rod_idx = value['pris_id']

            if 'W' in name:  # Joints for black are mirrored so signs are needed
                mask = self.white_rods_mask[:, rod_idx]
                one_hot_encoding[mask, 0] = 1
            elif 'B' in name:
                mask = self.black_rods_mask[:, rod_idx-4]
                one_hot_encoding[mask, 1] = 1
            fig_obs = torch.cat((one_hot_encoding, fig_tpos), dim=1)
            obj_obs.append(fig_obs)
        obs = torch.cat(obj_obs, dim=-1).transpose(1, 2)

        # Select random white figure as target to spawn ball
        obs = obs[env_ids]
        active_figures = torch.sum(obs[..., :2], dim=-1).to(torch.int32)
        num_figures = torch.sum(active_figures, dim=-1)
        choice = randint(num_figures, size=num_figures.shape).to(self.device)
        idx = torch.cumsum(num_figures, dim=0) - num_figures + choice
        xy = obs[active_figures == 1][idx][:, -2:]

        # True if black, False if white. With "*2-1" we get -1 for all whites
        sign = (obs[active_figures == 1][idx][:, 1] == 1) * 2 - 1

        init_ball_pos = self._init_ball_position[env_ids].clone()
        init_ball_pos[:, :2] = xy  # Move ball to figure
        init_ball_pos[:, 0] += sign * 4e-2  # Move ball in front of figure
        init_ball_pos[:, 1] += 2e-2 * (2 * torch.rand(num_resets, device=self.device) - 1)  # Add noise to y pos

        init_ball_rot = self._init_ball_rotation[env_ids].clone()
        self._balls.set_world_poses(init_ball_pos + self._env_pos[env_ids], init_ball_rot, indices=indices)

        init_ball_vel = torch.zeros_like(self._init_ball_velocities[env_ids])
        self._balls.set_velocities(init_ball_vel, indices=indices)

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
            joint_targets = self._default_joint_pos[:, self.active_joint_dofs].clone()
            joint_targets[self.active_dof_mask] = targets[self.active_dof_mask]

        joint_targets = self.get_delayed_joint_actions(joint_targets)
        self.joint_targets = joint_targets.clone()

        if self.low_level_controller is not None:
            self.low_level_controller.set_target(joint_targets)
        else:
            self.apply_control_target(joint_targets)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
