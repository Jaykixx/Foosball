from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.torch.maths import *
from utils.robots.foosball import Foosball
import omni.replicator.core as rep
from pxr import PhysxSchema
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

        # Check if video capture is required
        self.headless = self._cfg["headless"]
        self.capture = self._cfg["capture"] if not self.headless else False
        self.frame_id = 0

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

    def set_up_scene(self, scene) -> None:
        self.get_game_table()
        self.get_game_ball()

        super().set_up_scene(scene)

        # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(get_prim_at_path('/physicsScene'))
        # physxSceneAPI.CreateEnableCCDAttr().Set(True)

        # Get robot articulations
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/env_.*/Foosball", name="robot_view", reset_xform_properties=False
        )
        scene.add(self._robots)

        # Get ball view
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/Ball", name="ball_view", reset_xform_properties=False
        )
        scene.add(self._balls)

    def get_game_table(self) -> None:
        self.robot = Foosball(self.default_zero_env_path, device=self.device)
        self._sim_config.apply_articulation_settings(
            "Foosball", self.robot.reference, self._sim_config.parse_actor_config("Foosball")
        )
        self.rev_joints = self.robot.dof_paths_rev
        self.pris_joints = self.robot.dof_paths_pris

    # def get_game_ball(self) -> None:
    #     ball_path = self.default_zero_env_path + "/Ball"
    #     self._ball_radius = 0.01725
    #     self._sim_config.apply_articulation_settings(
    #         "Ball", get_prim_at_path(ball_path),
    #         self._sim_config.parse_actor_config("Ball")
    #     )

    def get_game_ball(self) -> None:
        physics_material_path = find_unique_string_name(
            initial_name="/World/Physics_Materials/physics_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        physics_material = PhysicsMaterial(
            prim_path=physics_material_path,
            dynamic_friction=0.2,
            static_friction=0.2,
            restitution=0.5,
        )

        ball_path = self.default_zero_env_path + "/Ball"
        self._init_ball_position = torch.tensor(
            [[0, 0, 0.79025]], device=self.device
        ).repeat(self.num_envs, 1)
        self._init_ball_rotation = torch.tensor(
            [[1, 0, 0, 0]], device=self.device
        ).repeat(self.num_envs, 1)
        self._init_ball_velocities = torch.zeros((self.num_envs, 6), device=self.device)
        self._ball_radius = 0.01725
        self.ball = DynamicSphere(
            prim_path=ball_path,
            radius=self._ball_radius,
            color=torch.tensor([255, 191, 0], device=self.device),
            name="ball_0",
            mass=0.023,
            physics_material=physics_material
        )
        self._sim_config.apply_articulation_settings(
            "Ball", get_prim_at_path(ball_path),
            self._sim_config.parse_actor_config("Ball")
        )
        # physx_rb_api = self._sim_config._get_physx_rigid_body_api(self.ball.prim)
        # physx_rb_api.CreateEnableCCDAttr().Set(True)

    def get_observations(self) -> dict:
        # Observe figurines
        fig_pos = self._robots.get_joint_positions(joint_indices=self.active_dofs, clone=False)
        fig_vel = self._robots.get_joint_velocities(joint_indices=self.active_dofs, clone=False)

        # Observe game ball in x-, y-axis
        ball_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_pos[:, :2] - self._env_pos[:, :2]
        ball_vel = self._balls.get_velocities(clone=False)[:, :2]

        # Rescale figure observations
        offset = self.dof_offset[..., self.active_dofs]
        range = self.dof_range[..., self.active_dofs]
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

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_resets = len(env_ids)

        # # Reset ball to randomized positions and velocities
        # init_ball_pos = self._init_ball_position[env_ids].clone()
        # init_ball_rot = self._init_ball_rotation[env_ids].clone()
        # init_ball_pos[..., 1] -= 0.3
        # self._balls.set_world_poses(init_ball_pos + self._env_pos[env_ids], init_ball_rot, indices=indices)
        #
        # init_ball_vel = self._init_ball_velocities[env_ids].clone()
        # init_ball_vel[..., 0] = torch_rand_float(-3, 3, (num_resets, 1), self._device).squeeze()
        # init_ball_vel[..., 1] = torch_rand_float(1, 5, (num_resets, 1), self._device).squeeze()
        # init_ball_vel[..., 2:] = 0
        # self._balls.set_velocities(init_ball_vel, indices=indices)

        # Signs determines which goal is targeted
        # print("Reseting Environments.")
        signs = torch.sign(torch.rand(num_resets, device=self.device) - 0.5)
        # signs = 1
        init_ball_pos = self._init_ball_position[env_ids].clone()
        init_ball_rot = self._init_ball_rotation[env_ids].clone()
        y_offset = torch_rand_float(-0.3, 0.3, (num_resets, 1), self._device).squeeze()
        init_ball_pos[..., 1] += signs * y_offset
        self._balls.set_world_poses(init_ball_pos + self._env_pos[env_ids], init_ball_rot, indices=indices)

        init_ball_vel = self._init_ball_velocities[env_ids]
        d1 = y_offset.abs() - 0.205 / 2 + 2 * self._ball_radius
        d2 = y_offset.abs() + 0.205 / 2 - 2 * self._ball_radius
        xd1 = torch.sqrt(25 / (1 + (d1 / 1.08) ** 2))
        xd2 = torch.sqrt(25 / (1 + (d2 / 1.08) ** 2))
        xd_min = torch.minimum(xd1, xd2)
        xd_max = torch.maximum(xd1, xd2)
        xd = torch.rand_like(xd_min, device=self._device) * (xd_max - xd_min) + xd_min
        yd = - torch.sign(y_offset) * torch.sqrt(25 - xd ** 2)
        init_ball_vel[..., 0] = signs * xd
        init_ball_vel[..., 1] = signs * yd
        init_ball_vel[..., 2:] = 0
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        self.actions = actions.clone().to(self.device)

        # self._robots.set_joint_velocity_targets(
        #     actions * self._robot_vel_limit[:, self.active_dofs], joint_indices=self.active_dofs
        # )

        targets = self.actions * self.dof_range[:, self.active_dofs]/2 + self.dof_offset[:, self.active_dofs]
        targets = torch.clamp(
            targets, self.robot_lower_limit[self.active_dofs], self.robot_upper_limit[self.active_dofs]
        )
        self._robots.set_joint_position_targets(targets, joint_indices=self.active_dofs)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def post_reset(self) -> None:
        if not hasattr(self, 'active_dofs'):
            self.active_dofs = list(range(self._robots.num_dof))
        if self.capture and not hasattr(self, 'rgb_annotators'):
            self.get_camera_sensor()

        self._robots.switch_control_mode('position')
        # self._robots.switch_control_mode('velocity')

        limits = self._robots.get_dof_limits().clone().to(device=self._device)
        self.robot_lower_limit = limits[0, :, 0]
        self.robot_upper_limit = limits[0, :, 1]
        self.dof_range = limits[..., 1] - limits[..., 0]
        self.dof_offset = (limits[..., 1] - limits[..., 0]) / 2 + limits[..., 0]

        rev_joints = {name: self._robots.get_dof_index(name) for name in self.rev_joints}
        inactive_rev_joints = [joint for joint in rev_joints.values() if joint not in self.active_dofs]

        pris_joints = {name: self._robots.get_dof_index(name) for name in self.pris_joints}
        self.active_pris_joints = {key: value for key, value in pris_joints.items() if value in self.active_dofs}

        self._default_joint_pos = self.dof_offset.clone()
        self._default_joint_pos[:, inactive_rev_joints] += np.pi/2
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

    def calculate_metrics(self) -> None:
        pos = self._balls.get_world_poses(clone=False)[0]
        pos = pos - self._env_pos

        # Compute distance ball to goal reward
        # dist_to_w_goal = torch.sqrt(torch.pow(pos[:, 0] - 0.62, 2) + torch.pow(pos[:, 1], 2))
        dist_to_b_goal = torch.sqrt(torch.pow(pos[:, 0] + 0.62, 2) + torch.pow(pos[:, 1], 2))

        dist_to_goal_rew = torch.exp(-dist_to_b_goal / 0.5) ** 3  # - torch.exp(-dist_to_w_goal / 0.5) ** 3
        dist_to_goal_rew = dist_to_goal_rew * 0  # torch.exp(-self.progress_buf / 100)

        # Regularization of actions
        # action_penalty_w = torch.sum(self.actions[..., :self._num_actions] ** 2, dim=-1)
        # # action_penalty_b = torch.sum(self.actions[..., self._num_actions:] ** 2, dim=-1)
        # action_penalty = 1e-2 * (action_penalty_w)  # - action_penalty_b)

        reward = dist_to_goal_rew  # - action_penalty

        # Compute distance of figures to ball in y-direction
        for joint, id in self.active_pris_joints.items():
            if joint in self.robot.dof_paths_W:
                joint_pos = self._robots.get_joint_positions(joint_indices=[id], clone=False)
                offsets = self.robot.figure_positions[joint.split('_')[0]]
                fig_pos = - joint_pos.repeat(1, len(offsets)) + offsets.unsqueeze(0)
                fig_pos_dist = torch.abs(fig_pos - pos[:, 1:2])
                fig_pos_dist = torch.min(fig_pos_dist, dim=-1)[0]
                fig_pos_rew = torch.exp(-fig_pos_dist / 0.08) * 1e-1
                reward += fig_pos_rew

        self.rew_buf = reward

        mask_y = torch.min(-0.08 < pos[:, 1], pos[:, 1] < 0.08)

        # Check white goal hit
        mask_x = 0.62 < pos[:, 0]
        loss_mask = torch.min(mask_x, mask_y)
        self.rew_buf[loss_mask] = -1000

        # Check black goal hit
        mask_x = pos[:, 0] < -0.62
        win_mask = torch.min(mask_x, mask_y)
        win_rew_mask = torch.min(win_mask, self.progress_buf > 12)
        self.rew_buf[win_rew_mask] = 1000

        # Check Termination penalty
        limit = self._init_ball_position[0, 2] + self.termination_height
        mask_z = pos[:, 2] > limit
        self.rew_buf[mask_z] = - self.termination_penalty

        # Check done flags
        goal_mask = torch.max(win_mask, loss_mask)
        length_mask = self.progress_buf >= self._max_episode_length - 1
        limit = self._init_ball_position[0, 2] + self.termination_height
        termination_mask = pos[:, 2] > limit
        self.reset_buf = torch.max(goal_mask, length_mask)
        self.reset_buf = torch.max(self.reset_buf, termination_mask)

        self.extras["battle_won"][:] = 0
        self.extras["battle_won"][win_mask] = 1
        self.extras["battle_won"][loss_mask] = -1
        self.extras["Games_finished_with_goals"] = goal_mask.sum() / self.reset_buf.sum()

    def is_done(self) -> None:
        # Is part of calculate_metrics for performance!
        pass

    def get_camera_sensor(self) -> None:
        self.frame_path = os.path.join(os.getcwd(), f"runs/{self.name}/capture")
        os.makedirs(self.frame_path, exist_ok=True)

        self.rgb_annotators = []
        for i in range(self.num_envs):
            camera_path = self.default_zero_env_path[:-1] + f"{i}/Foosball/Top_Down_Camera"
            rp = rep.create.render_product(camera_path, resolution=(1280, 720))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([rp])
            self.rgb_annotators.append(rgb)

    def capture_image(self):
        for i, rgb_annotator in enumerate(self.rgb_annotators):
            data = rgb_annotator.get_data()
            if len(data) > 0:  # Catch cases after reset
                path = self.frame_path + f"/rp_{i}_frame_" + str(self.frame_id).zfill(3)
                save_rgb(data, path)
        self.frame_id += 1


def save_rgb(rgb_data, file_name):
    # Save rgb image to file
    rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
    rgb_img = Image.fromarray(rgb_image_data, "RGBA")
    rgb_img.save(file_name + ".png")
