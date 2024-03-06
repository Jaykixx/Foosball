from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.kit.viewport.utility import get_viewport_from_window_name
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from omni.isaac.core.utils.torch.maths import *
from utilities.robots.foosball import Foosball
from utilities.models.kalman_filter import KalmanFilter
import omni.replicator.core as rep
from pxr import PhysxSchema, UsdPhysics, Usd, UsdGeom
from PIL import Image
import numpy as np
import torch
import os
import time


class FoosballTask(RLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # Environment grid parameters
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # Environment constraints
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        # Termination conditions
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.termination_penalty = self._task_cfg["env"]["terminationPenalty"]

        phys_dt = self._sim_config.sim_params["dt"]
        self.control_frequency_inv = self._task_cfg["env"]["controlFrequencyInv"]
        self.dt = phys_dt * self.control_frequency_inv

        if not hasattr(self, "_num_observations"):
            self._num_observations = 36
        if not hasattr(self, "_num_actions"):
            self._num_actions = 16
        if not hasattr(self, "_dof"):
            self._dof = 16

        super(FoosballTask, self).__init__(name, env, offset)

        # Check if video capture is required
        self.headless = self._cfg["headless"]
        self.capture = self._cfg["capture"] if not self.headless else False
        self.frame_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.game_counter = torch.ones(self.num_envs, device=self.device, dtype=torch.int32) * -2

        if not hasattr(self, "kalman"):
            self.apply_kalman_filter = self._task_cfg["env"].get("applyKalmanFiltering", False)
            if self.apply_kalman_filter:
                n_obs = self._dof - self.num_actions + 2  # Only on uncontrolled rods and ball
                self.kalman = KalmanFilter(2 * n_obs, n_obs, self.num_envs, self._device)

        self.observations_dofs = []
        self.old_actions = torch.zeros((self.num_envs, self._dof), device=self._device)

    def set_initial_camera_params(self, camera_position=(0, 0, 10),
                                  camera_target=(0, 0, 0)):
        if not self.headless:
            cam_prim_path = f"/World/envs/env_0/Foosball/Top_Down_Camera"
            viewport_api_2 = get_viewport_from_window_name("Viewport")
            viewport_api_2.set_active_camera(cam_prim_path)
        else:
            super().set_initial_camera_params(camera_position=camera_position,
                                              camera_target=camera_target)

    def create_motion_capture_camera(self):
        cam_prim_path = f"/World/envs/env_0/Foosball/Top_Down_Camera"
        camera_prim = get_prim_at_path(cam_prim_path)
        physxRbAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(camera_prim)
        physxRbAPI.CreateDisableGravityAttr().Set(True)
        self.camera = RigidPrimView(cam_prim_path)

        cam_pos = self.camera.get_world_poses(clone=False)[0]
        cam_pos = cam_pos - self._env_pos[0:1] + self._env_pos[256:257]
        self.camera.set_world_poses(cam_pos)

        lin_vel = torch.tensor([[0, -0.1, 0.1]])
        ang_vel = torch.tensor([[18, 0, 0]])
        cam_vel = torch.concatenate((lin_vel, ang_vel), dim=-1)
        self.camera.set_velocities(cam_vel)

    def set_up_scene(self, scene) -> None:
        self.get_game_table()
        self.get_game_ball()

        super().set_up_scene(scene)

        # Get robot articulations
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/env_.*/Foosball", name="robot_view", reset_xform_properties=False
        )
        scene.add(self._robots)

        # Get ball view
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/Foosball/Ball", name="ball_view", reset_xform_properties=False
        )
        scene.add(self._balls)

        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(get_prim_at_path('/physicsScene'))
        physxSceneAPI.CreateEnableCCDAttr().Set(True)

        # self.create_motion_capture_camera()

    def get_game_table(self) -> None:
        self.robot = Foosball(self.default_zero_env_path, device=self.device)
        self._sim_config.apply_articulation_settings(
            "Foosball", self.robot.reference, self._sim_config.parse_actor_config("Foosball")
        )
        self.rev_joints = self.robot.dof_paths_rev
        self.pris_joints = self.robot.dof_paths_pris

    def get_game_ball(self) -> None:
        ball_path = self.default_zero_env_path + "/Foosball/Ball"
        self._init_ball_position = torch.tensor(
            [[0, 0, 0.79025]], device=self.device
        ).repeat(self.num_envs, 1)
        self._init_ball_rotation = torch.tensor(
            [[1, 0, 0, 0]], device=self.device
        ).repeat(self.num_envs, 1)
        self._init_ball_velocities = torch.zeros((self.num_envs, 6), device=self.device)
        self._ball_radius = 0.01725

        self._sim_config.apply_articulation_settings(
            "Ball", get_prim_at_path(ball_path),
            self._sim_config.parse_actor_config("Ball")
        )

        # physx_rb_api = self._sim_config._get_physx_rigid_body_api(get_prim_at_path(ball_path))
        # physx_rb_api.CreateEnableCCDAttr().Set(True)

    def get_observations(self) -> dict:
        # Observe figurines
        fig_pos = self._robots.get_joint_positions(joint_indices=self.observations_dofs, clone=False)
        fig_vel = self._robots.get_joint_velocities(joint_indices=self.active_dofs, clone=False)

        # Observe game ball in x-, y-axis
        ball_obs = self._balls.get_world_poses(clone=False)[0]
        ball_obs = ball_obs[:, :2] - self._env_pos[:, :2]

        if self.apply_kalman_filter:
            self.kalman.predict()
            kstate = self.kalman.state.clone()
            ball_pos, ball_vel = kstate[:, :2, 0], kstate[:, 2:, 0] * 60
            self.kalman.correct(ball_obs.unsqueeze(-1))
        else:
            ball_vel = self._balls.get_velocities(clone=False)[:, :2]
            ball_pos = ball_obs

        # Rescale figure observations
        offset = self.dof_offset[..., self.observations_dofs]
        range = self.dof_range[..., self.observations_dofs]
        fig_pos = 2 * (fig_pos - offset) / range
        fig_vel = fig_vel / self._robot_vel_limit[..., self.active_dofs]

        self.obs_buf = torch.cat(
            (fig_pos, fig_vel, ball_pos, ball_vel), dim=-1
        )

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }

        if self.capture:
            self.capture_image()
        return observations

    def get_robot(self):
        return self._robots

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        # Reset joint positions and velocities
        self._robots.set_joint_position_targets(
            self._default_joint_pos[env_ids], indices=indices
        )
        self._robots.set_joint_positions(
            self._default_joint_pos[env_ids], indices=indices
        )
        self._robots.set_joint_velocities(
            self._default_joint_vel[env_ids], indices=indices
        )

        self.reset_ball(env_ids)

        self.game_counter[env_ids] += 1
        self.frame_id[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_resets = len(env_ids)

        # Reset ball to randomized positions and velocities
        sign = torch.sign(torch.rand(num_resets, device=self.device) - 0.5)
        init_ball_pos = self._init_ball_position[env_ids].clone()
        init_ball_rot = self._init_ball_rotation[env_ids].clone()
        init_ball_pos[..., 1] -= sign * 0.3
        self._balls.set_world_poses(init_ball_pos + self._env_pos[env_ids], init_ball_rot, indices=indices)

        init_ball_vel = self._init_ball_velocities[env_ids].clone()
        xvel = torch_rand_float(0.25, 1, (num_resets, 1), self._device).squeeze()
        xsign = torch.sign(torch.rand(num_resets, device=self.device) - 0.5)
        init_ball_vel[..., 0] = xsign * xvel
        init_ball_vel[..., 1] = sign * torch_rand_float(1, 2, (num_resets, 1), self._device).squeeze()
        init_ball_vel[..., 2:] = 0
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        self.actions = actions.clone().to(self.device)
        targets = self.actions * self.dof_range[:, self.active_dofs]/2 + self.dof_offset[:, self.active_dofs]
        targets = torch.clamp(
            targets, self.robot_lower_limit[self.active_dofs], self.robot_upper_limit[self.active_dofs]
        )
        self._robots.set_joint_position_targets(targets, joint_indices=self.active_dofs)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def hide_inactive_rods(self):
        flags = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for rod_name in self.inactive_rods:
            if rod_name[-1] == 'W':
                rod_prim = XFormPrimView(
                    prim_paths_expr="/World/envs/env_.*/Foosball/White/" + rod_name,
                    name="rod_view",
                    reset_xform_properties=False
                )
            else:
                rod_prim = XFormPrimView(
                    prim_paths_expr="/World/envs/env_.*/Foosball/Black/" + rod_name,
                    name="rod_view",
                    reset_xform_properties=False
                )
            rod_prim.set_visibilities(flags)

    def post_reset(self) -> None:
        if not hasattr(self, 'active_dofs'):
            self.active_dofs = list(range(self._robots.num_dof))
        if self.capture and not hasattr(self, 'rgb_annotators'):
            self.get_camera_sensor()

        self.observations_dofs += self.active_dofs

        self._robots.switch_control_mode('position')

        limits = self._robots.get_dof_limits().clone().to(device=self._device)
        self.robot_lower_limit = limits[0, :, 0]
        self.robot_upper_limit = limits[0, :, 1]
        self.dof_range = limits[..., 1] - limits[..., 0]
        self.dof_offset = (limits[..., 1] - limits[..., 0]) / 2 + limits[..., 0]

        rev_joints = {name: self._robots.get_dof_index(name) for name in self.rev_joints}
        self.inactive_rods = {}
        for joint_name, joint_id in rev_joints.items():
            if joint_id not in self.observations_dofs:
                rod_name = '_'.join(joint_name.split('_')[:2])
                partner_joint = rod_name + "_PrismaticJoint"
                partner_joint_id = self._robots.get_dof_index(partner_joint)
                if partner_joint_id not in self.observations_dofs:
                    self.inactive_rods[rod_name] = joint_id

        self.hide_inactive_rods()

        pris_joints = {name: self._robots.get_dof_index(name) for name in self.pris_joints}
        self.active_pris_joints = {key: value for key, value in pris_joints.items() if value in self.active_dofs}

        self._default_joint_pos = self.dof_offset.clone()
        self._default_joint_pos[:, list(self.inactive_rods.values())] += np.pi/2
        self._default_joint_vel = torch.zeros_like(self._default_joint_pos)

        self._robot_vel_limit = self.robot.qdlim[None]

        # randomize all envs
        indices = torch.arange(
            self._robots.count, dtype=torch.int64, device=self._device
        )
        self.reset_idx(indices)

        self.extras["battle_won"] = torch.zeros(self._num_envs,
                                                device=self._device,
                                                dtype=torch.long)

    def _compute_action_regularization(self):
        # Regularization of actions
        action_diff = self.actions - self.old_actions
        action_penalty_w = torch.mean(action_diff[..., :self._num_actions] ** 2, dim=-1)
        return - action_penalty_w

    def _compute_ball_to_goal_distances(self, ball_pos):
        # Compute distance ball to goal reward
        z = torch.zeros_like(ball_pos[:, 1])
        y_dist = torch.pow(torch.max(torch.abs(ball_pos[:, 1]) - 0.08525, z), 2)
        x_dist_to_b_goal = torch.pow(ball_pos[:, 0] + 0.61725, 2)
        x_dist_to_w_goal = torch.pow(ball_pos[:, 0] - 0.61725, 2)
        dist_to_w_goal = torch.sqrt(x_dist_to_w_goal + y_dist)
        dist_to_b_goal = torch.sqrt(x_dist_to_b_goal + y_dist)

        return dist_to_b_goal, dist_to_w_goal

    def _compute_fig_to_ball_distances(self, ball_pos):
        # Compute distance of figures to ball in y-direction
        distances = []
        for joint, id in self.active_pris_joints.items():
            if joint in self.robot.dof_paths_W:
                joint_pos = self._robots.get_joint_positions(joint_indices=[id], clone=False)
                offsets = self.robot.figure_positions[joint.split('_')[0]]
                fig_pos = - joint_pos.repeat(1, len(offsets)) + offsets.unsqueeze(0)
                fig_pos_dist = torch.abs(fig_pos - ball_pos[:, 1:2])
                distances.append(torch.min(fig_pos_dist, dim=-1)[0])
        return distances

    def _dist_to_goal_reward(self, ball_pos):
        dist_to_b_goal, dist_to_w_goal = self._compute_ball_to_goal_distances(ball_pos)
        dist_to_goal_rew = torch.exp(-6 * dist_to_b_goal)  # - torch.exp(-6*dist_to_w_goal)
        return dist_to_goal_rew

    def _fig_to_ball_reward(self, ball_pos):
        fig_pos_dist = torch.stack(self._compute_fig_to_ball_distances(ball_pos))
        fig_pos_rew = - (1 - torch.exp(-6 * fig_pos_dist).mean(0))
        return fig_pos_rew

    def _calculate_metrics(self, ball_pos) -> None:
        self.rew_buf[:] = 0

        mask_y = torch.min(-0.0925 < ball_pos[:, 1], ball_pos[:, 1] < 0.0925)

        # Check white goal hit
        mask_x = 0.61725 < ball_pos[:, 0]
        loss_mask = torch.min(mask_x, mask_y)
        self.rew_buf[loss_mask] = -1000

        # Check black goal hit
        mask_x = ball_pos[:, 0] < -0.61725
        win_mask = torch.min(mask_x, mask_y)
        self.rew_buf[win_mask] = 1000

        # Check Termination penalty
        limit = self._init_ball_position[0, 2] + self.termination_height
        mask_z = ball_pos[:, 2] > limit
        self.rew_buf[mask_z] = - self.termination_penalty

        # Check done flags
        goal_mask = torch.max(win_mask, loss_mask)
        length_mask = self.progress_buf >= self._max_episode_length - 1
        self.reset_buf = torch.max(goal_mask, length_mask)
        self.reset_buf = torch.max(self.reset_buf, mask_z)

        self.extras["battle_won"][:] = 0
        self.extras["battle_won"][win_mask] = 1
        self.extras["battle_won"][loss_mask] = -1
        if self.reset_buf.sum() > 0:
            self.extras["Games_finished_with_goals"] = goal_mask.sum() / self.reset_buf.sum()
            self.extras["Win_Rate"] = win_mask.sum() / self.reset_buf.sum()
            self.extras["Loss_Rate"] = loss_mask.sum() / self.reset_buf.sum()
        else:
            self.extras["Games_finished_with_goals"] = 0
            self.extras["Win_Rate"] = 0
            self.extras["Loss_Rate"] = 0

        # Counting Glitches for debugging purposes
        debug_cond_x = torch.max(ball_pos[:, 0] < -0.71, ball_pos[:, 0] > 0.71)
        debug_cond_y = torch.max(ball_pos[:, 1] < -0.367, ball_pos[:, 1] > 0.367)
        debug_cond_z = ball_pos[:, 2] < 0.7
        debug_cond = torch.max(torch.max(debug_cond_x, debug_cond_y), debug_cond_z)
        debug_cond[goal_mask] = 0
        self.reset_buf = torch.max(self.reset_buf, debug_cond)
        self.extras["Ended in Glitch"] = debug_cond.sum()

    def calculate_metrics(self) -> None:
        pos = self._balls.get_world_poses(clone=False)[0]
        pos = pos - self._env_pos

        self._calculate_metrics(pos)

        self.old_actions = self.actions

    def is_done(self) -> None:
        # Is part of calculate_metrics for performance!
        pass

    def get_camera_sensor(self) -> None:
        self.rgb_annotators = []
        self.frame_paths = []
        for i in range(self.num_envs):
            frame_path = os.path.join(os.getcwd(), f"runs/{self.name}/capture/Env_{i}")
            os.makedirs(frame_path, exist_ok=True)
            self.frame_paths.append(frame_path)
            camera_path = self.default_zero_env_path[:-1] + f"{i}/Foosball/Top_Down_Camera"
            rp = rep.create.render_product(camera_path, resolution=(1280, 720))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([rp])
            self.rgb_annotators.append(rgb)

    def capture_image(self):
        for i, rgb_annotator in enumerate(self.rgb_annotators):
            data = rgb_annotator.get_data()
            if len(data) > 0:  # Catch cases after reset
                frame_path = os.path.join(self.frame_paths[i], f"Game_{self.game_counter[i].item():02}")
                if not os.path.isdir(frame_path):
                    os.makedirs(frame_path, exist_ok=True)
                path = frame_path + f"/frame_{self.frame_id[i].item():04}"
                save_rgb(data, path)
            self.frame_id[i] += 1


def save_rgb(rgb_data, file_name):
    # Save rgb image to file
    rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
    rgb_img = Image.fromarray(rgb_image_data, "RGBA")
    rgb_img.save(file_name + ".png")
