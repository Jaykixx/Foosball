from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects.sphere import DynamicSphere
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView
from utils.robots.foosball import Foosball
import omni.replicator.core as rep
from pxr import PhysxSchema
from PIL import Image
import numpy as np
import torch
import os


class FoosballTask(RLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # Check if video capture is required
        headless = self._cfg["headless"]
        self.capture = self._cfg["capture"] if not headless else False
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

        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(get_prim_at_path('/physicsScene'))
        physxSceneAPI.CreateEnableCCDAttr().Set(True)

        # Get robot articulations
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/env_.*/Foosball", name="robot_view"
        )
        scene.add(self._robots)

        # Get ball view
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/Ball", name="ball_view"
        )
        scene.add(self._balls)

    def get_game_table(self) -> None:
        self.robot = Foosball(self.default_zero_env_path, device=self.device)
        self._sim_config.apply_articulation_settings(
            "Foosball", self.robot.reference, self._sim_config.parse_actor_config("Foosball")
        )
        self.rev_joints = self.robot.dof_paths_rev

    def get_game_ball(self) -> None:
        ball_path = self.default_zero_env_path + "/Ball"
        self._init_ball_position = torch.tensor(
            [[0, 0, 0.79025]], device=self.device
        ).repeat(self._num_envs, 1)
        self._ball_radius = 0.01725
        self._sim_config.apply_articulation_settings(
            "Ball", get_prim_at_path(ball_path),
            self._sim_config.parse_actor_config("Ball")
        )

    def get_observations(self) -> dict:
        # Observe figurines (only white)
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

        self.obs_buf[:] = torch.cat(
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
        self._robots.set_joint_positions(
            self._default_joint_pos[env_ids], indices=indices
        )
        self._robots.set_joint_velocities(
            self._default_joint_vel[env_ids], indices=indices
        )

        # Reset ball to randomized position
        self.reset_ball(env_ids)

        self.progress_buf[env_ids] = 0

    def reset_ball(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        init_ball_pos = self._init_ball_position[env_ids].clone()
        # init_ball_pos[..., 1] += 0.3
        self._balls.set_world_poses(
            init_ball_pos + self._env_pos[env_ids], indices=indices
        )

        yd = - torch.rand(len(indices)) * 5
        xd = (torch.rand(len(indices)) * 2 - 1) * 2
        z = torch.zeros_like(xd)
        init_ball_vel = torch.stack((xd, yd, z, z, z, z), dim=-1)
        self._balls.set_velocities(init_ball_vel, indices=indices)

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # self.actions = actions.clone().to(self.device)

        # self._robots.set_joint_velocity_targets(
        #     actions * self._robot_vel_limit[:, self.active_dofs], joint_indices=self.active_dofs
        # )

        actions = actions * self.dof_range[:, self.active_dofs]/2 + self.dof_offset[:, self.active_dofs]
        actions = torch.clamp(
            actions, self.robot_lower_limit[self.active_dofs], self.robot_upper_limit[self.active_dofs]
        )
        self._robots.set_joint_position_targets(actions, joint_indices=self.active_dofs)

    def post_reset(self) -> None:
        if not hasattr(self, 'active_dofs'):
            self.active_dofs = list(range(self._robots.num_dof))
        if self.capture and not hasattr(self, 'rgb_annotators'):
            self.get_camera_sensor()

        self._robots.switch_control_mode('position')
        # self._robots.switch_control_mode('velocity')

        limits = self._robots.get_dof_limits().to(device=self._device)
        self.robot_lower_limit = limits[0, :, 0]
        self.robot_upper_limit = limits[0, :, 1]
        self.dof_range = limits[..., 1] - limits[..., 0]
        self.dof_offset = (limits[..., 1] - limits[..., 0]) / 2 + limits[..., 0]

        rev_joints = [self._robots.get_dof_index(name) for name in self.rev_joints]
        self._default_joint_pos = self.dof_offset.clone()
        self._default_joint_pos[:, rev_joints] += np.pi/2
        self._default_joint_vel = torch.zeros_like(self._default_joint_pos)

        self._robot_vel_limit = self.robot.qdlim[None]

        # randomize all envs
        indices = torch.arange(
            self._robots.count, dtype=torch.int64, device=self._device
        )
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        pos = self._balls.get_world_poses(clone=False)[0]
        pos = pos - self._env_pos

        mask_y = torch.min(-0.08 < pos[:, 1], pos[:, 1] < 0.08)

        # Check white goal hit
        mask_x = 0.62 < pos[:, 0]
        loss_mask = torch.min(mask_x, mask_y)
        self.rew_buf[loss_mask] = -1

        # Check black goal hit
        mask_x = pos[:, 0] < -0.62
        win_mask = torch.min(mask_x, mask_y)
        self.rew_buf[win_mask] = 1

        # Neither win or loss
        neutral_mask = ~torch.max(win_mask, loss_mask)
        self.rew_buf[neutral_mask] = 0

        # Check Termination penalty
        # limit = self._init_ball_position[0, 2] + self.termination_height
        # mask_z = pos[:, 2] > limit
        # self.rew_buf[mask_z] = - self.termination_penalty

    def is_done(self) -> None:
        pos = self._balls.get_world_poses(clone=False)[0]
        pos = pos - self._env_pos

        # Check goal hit
        mask_x = torch.max(pos[:, 0] < -0.62, 0.62 < pos[:, 0])
        mask_y = torch.min(-0.08 < pos[:, 1], pos[:, 1] < 0.08)
        goal_mask = torch.min(mask_x, mask_y)

        # Check if ball exceeds height limit
        limit = self._init_ball_position[0, 2] + self.termination_height
        mask_z = pos[:, 2] > limit
        goal_mask = torch.max(goal_mask, mask_z)

        # Check for episode length
        length_mask = self.progress_buf >= self._max_episode_length
        self.reset_buf = torch.max(goal_mask, length_mask)

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
