from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import *
import omni.replicator.core as rep
from pxr import PhysxSchema

from utilities.models.low_level_controllers.polynomial_s_curve import SCurve
from utilities.models.kalman_filter import KalmanFilter
from utilities.robots.foosball import Foosball
from environments.base_task import BaseTask

from PIL import Image
import numpy as np
import torch
import os


class FoosballTask(BaseTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_actions"):
            # Defines action space for AI
            self._num_actions = 16
        if not hasattr(self, "_dof"):
            # Defines action space for task - Only different for selfplay
            self._dof = self._num_actions
        if not hasattr(self, "_num_task_observations"):
            self._num_task_observations = 4
        if not hasattr(self, "_num_joint_observations"):
            # Gripper observed as single boolean joint
            self._num_joint_observations = 2 * self._num_actions
        if not hasattr(self, "_num_observations"):
            self._num_observations = self._num_joint_observations + self._num_task_observations
        if not hasattr(self, "_num_objects"):
            # Number of involved figurines + ball
            self._num_objects = 23
        if not hasattr(self, "_num_object_types"):
            # White, Black & Ball
            self._num_obj_types = 3
        if not hasattr(self, "_num_obs_per_object"):
            # Joints/Figurines: Pos, Rot + corresponding velocities
            # Ball: X, Y Pos + corresponding velocities
            self._num_obj_features = 6

        super(FoosballTask, self).__init__(name, sim_config, env, offset)

        # Termination conditions
        self.termination_height = self._env_cfg["terminationHeight"]
        self.termination_penalty = self._env_cfg["terminationPenalty"]

        # Win and Loss Rewards
        self.win_reward = self._env_cfg["winReward"]
        self.loss_penalty = self._env_cfg["lossPenalty"]

        if not hasattr(self, "kalman"):
            self.apply_kalman_filter = self._env_cfg.get("applyKalmanFiltering", False)
        self.initialize_kalman_filter()

        self.observed_dofs = []
        self.active_joint_dofs = []
        self.passive_joint_dofs = []

        self._applyKinematicConstraints = self._env_cfg.get("applyKinematicConstraints", False)
        self.scurve_planner = None

    @property
    def applyKinematicContraints(self):
        return self._applyKinematicConstraints and (self.scurve_planner is not None)

    def initialize_kalman_filter(self):
        if self.apply_kalman_filter:
            n_obs = self._dof - self.num_actions + 2  # Only on uncontrolled rods and ball
            self.kalman = KalmanFilter(n_obs, self.num_envs, self._device)

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
        self.get_robot()
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
        
        # self.create_motion_capture_camera()
        
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(get_prim_at_path('/physicsScene'))
        physxSceneAPI.CreateEnableCCDAttr().Set(True)

        # Correct light settings
        if self._env.render_enabled:
            light = get_prim_at_path("/World/defaultDistantLight")
            light.GetAttribute('inputs:intensity').Set(1000)

    def get_robot(self) -> None:
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

    def reset_idx(self, env_ids):
        self.reset_ball(env_ids)
        BaseTask.reset_idx(self, env_ids)
        if self.applyKinematicContraints:
            self.scurve_planner.a0[:] = 0

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

    def get_observations(self) -> dict:
        # Observe Joints
        dof_pos = self._robots.get_joint_positions(joint_indices=self.active_joint_dofs, clone=False)
        dof_vel = self._robots.get_joint_velocities(joint_indices=self.active_joint_dofs, clone=False)

        # Observe Figurines
        tpos = dof_pos.view(self.num_envs, 2, -1)[:, 0]
        rpos = dof_pos.view(self.num_envs, 2, -1)[:, 1]

        tvel = dof_vel.view(self.num_envs, 2, -1)[:, 0]
        rvel = dof_vel.view(self.num_envs, 2, -1)[:, 1]

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

        if len(self.passive_joint_dofs):
            passive_fig_pos = self._robots.get_joint_positions(joint_indices=self.passive_joint_dofs, clone=False)
            self.obs_buf = torch.cat(
                (dof_pos, dof_vel, passive_fig_pos, ball_pos, ball_vel), dim=-1
            )
        else:
            self.obs_buf = torch.cat(
                (dof_pos, dof_vel, ball_pos, ball_vel), dim=-1
            )

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf,
                "oc_obs_buf": self.oc_obs_buf
            }
        }

        if self.capture:
            self.capture_image()
        return observations

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

    # def set_gains(self, kps, kds):
    #     kps = torch.ones(16, device=self.device) * kps
    #     kds = torch.ones(16, device=self.device) * kds
    #     self._robots.set_gains(kps=kps, kds=kds, save_to_usd=False)

    def post_reset(self) -> None:
        BaseTask.post_reset(self)

        self.observed_dofs += self.active_joint_dofs

        rev_joints = {name: self._robots.get_dof_index(name) for name in self.rev_joints}
        self.inactive_rods = {}
        for joint_name, joint_id in rev_joints.items():
            if joint_id not in self.observed_dofs:
                rod_name = '_'.join(joint_name.split('_')[:2])
                partner_joint = rod_name + "_PrismaticJoint"
                partner_joint_id = self._robots.get_dof_index(partner_joint)
                if partner_joint_id not in self.observed_dofs:
                    self.inactive_rods[rod_name] = joint_id

        self.hide_inactive_rods()

        pris_joints = {name: self._robots.get_dof_index(name) for name in self.pris_joints}
        self.active_pris_joints = {key: value for key, value in pris_joints.items() if value in self.active_joint_dofs}

        self._default_joint_pos[:, list(self.inactive_rods.values())] += np.pi/2

        if self._applyKinematicConstraints:
            self.scurve_planner = SCurve(self.num_envs, self._dof, self.device)
            vmax = self.robot.qdlim[self.active_joint_dofs].expand(self.num_envs, -1)
            amax = self.robot.qddlim[self.active_joint_dofs].expand(self.num_envs, -1)
            jmax = self.robot.qdddlim[self.active_joint_dofs].expand(self.num_envs, -1)
            self.scurve_planner.set_limits(vmax, amax, jmax)
            self.set_low_level_controller(self.scurve_planner)
        # self.set_gains(100_000, 4_000)

        # randomize all envs
        indices = torch.arange(
            self._robots.count, dtype=torch.int64, device=self._device
        )
        self.reset_idx(indices)

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

    def _calculate_metrics(self):
        pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = pos - self._env_pos

        self.rew_buf[:] = 0

        mask_y = torch.min(-0.0925 < ball_pos[:, 1], ball_pos[:, 1] < 0.0925)

        # Check white goal hit
        mask_x = 0.61725 < ball_pos[:, 0]
        losses = torch.min(mask_x, mask_y)
        self.rew_buf[losses] = - self.loss_penalty

        # Check black goal hit
        mask_x = ball_pos[:, 0] < -0.61725
        wins = torch.min(mask_x, mask_y)
        # win_rew_mask = torch.min(wins, self.progress_buf > 12)
        self.rew_buf[wins] = self.win_reward

        # Check Termination penalty
        limit = self._init_ball_position[0, 2] + self.termination_height
        termination_mask = ball_pos[:, 2] > limit
        self.rew_buf[termination_mask] = - self.termination_penalty

        # Check done flags
        goal_mask = torch.max(wins, losses)
        timeouts = self.progress_buf >= self._max_episode_length - 1
        self.reset_buf = torch.max(goal_mask, timeouts)
        self.reset_buf = torch.max(self.reset_buf, termination_mask)

        return wins, losses, timeouts

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
