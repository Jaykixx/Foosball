from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.torch import *
from collections import deque
from PIL import Image
from abc import abstractmethod
from gym import spaces
import numpy as np
import torch
import os


class BaseTask(RLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._env_cfg = self._task_cfg["env"]

        # Environment parameters
        self._num_envs = self._env_cfg["numEnvs"]
        self._env_spacing = self._env_cfg["envSpacing"]
        self._max_episode_length = self._env_cfg["episodeLength"]
        self.max_vel_scale = self._env_cfg.get("max_joint_velocity_scale", 1)

        # Time step definition
        self.phys_dt = self._sim_config.sim_params["dt"]
        self.control_frequency_inv = self._env_cfg["controlFrequencyInv"]

        # Delayed actions
        if 'delayed_actions' in self._env_cfg:
            self.delay_actions = self._env_cfg["delayed_actions"]["active"]
            self.delayed_steps = self._env_cfg["delayed_actions"]["steps"]
        else:
            self.delay_actions = False

        self.anchored = False

        self.object_centric_obs = self._env_cfg['object_centric_obs']
        if self.object_centric_obs:
            obs_features = self._num_obj_types + self._num_obj_features
            self._num_observations = self._num_objects * obs_features
            self.observation_space = spaces.Box(
                np.ones((self._num_objects, obs_features), dtype=np.float32) * -1,
                np.ones((self._num_objects, obs_features), dtype=np.float32) * 1,
            )

        RLTask.__init__(self, name, env, offset)

        # For more flexibility when implementing hierarchical models etc.
        if not hasattr(self, "joint_observation_space"):
            self.joint_observation_space = spaces.Box(
                np.ones(self._num_joint_observations, dtype=np.float32) * -np.Inf,
                np.ones(self._num_joint_observations, dtype=np.float32) * np.Inf,
            )
        if not hasattr(self, "task_observation_space"):
            self.task_observation_space = spaces.Box(
                np.ones(self._num_task_observations, dtype=np.float32) * -np.Inf,
                np.ones(self._num_task_observations, dtype=np.float32) * np.Inf,
            )

        # Reset parameters
        self.joint_noise = self._env_cfg["resetJointNoise"]

        # Check if video capture is required
        self.headless = self._cfg["headless"]
        self.capture = self._cfg["capture"] if not self.headless else False
        self.frame_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.trial_counter = torch.ones(self.num_envs, device=self.device, dtype=torch.int32) * -2

        self.old_actions = torch.zeros((self.num_envs, self._dof), device=self._device)
        self.last_rew = torch.zeros_like(self.rew_buf)

        self.low_level_controller = None
        self.control_mode = 'position'
        self.target_mode = 'position'

    @property
    def dt(self):
        return self.phys_dt * self.control_frequency_inv

    def cleanup(self) -> None:
        RLTask.cleanup(self)
        if self.object_centric_obs:
            self.obs_buf = torch.zeros(
                (self._num_envs, self._num_objects, self._num_obj_types + self._num_obj_features),
                device=self._device, dtype=torch.float
            )

    def set_up_scene(self, scene, **kwargs) -> None:
        RLTask.set_up_scene(self, scene, **kwargs)
        self.num_joint_dof = self.robot.joint_dof
        self.joint_space = spaces.Box(
            self.robot.qlim[0].cpu().numpy(),
            self.robot.qlim[1].cpu().numpy(),
        )

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        default_joint_pos = self._default_joint_pos[:, self.active_joint_dofs].clone()
        default_joint_vel = self._default_joint_vel[:, self.active_joint_dofs].clone()

        joint_offsets = torch.rand(
            (self.num_envs, len(self.active_joint_dofs)), device=self._device
        )
        default_joint_pos = tensor_clamp(
            default_joint_pos + self.joint_noise * 2 * (joint_offsets - 0.5),
            self.joint_limits[:, self.active_joint_dofs, 0],
            self.joint_limits[:, self.active_joint_dofs, 1],
        )

        # Reset joint positions and velocities
        if hasattr(self, '_robots_dof_targets'):
            self._robots_dof_targets[env_ids] = default_joint_pos[env_ids].clone()
        self._robots.set_joint_positions(
            default_joint_pos[env_ids], indices=indices, joint_indices=self.active_joint_dofs
        )
        self._robots.set_joint_velocities(
            default_joint_vel[env_ids], indices=indices, joint_indices=self.active_joint_dofs
        )

        if self.delay_actions:
            if self.target_mode == 'velocity':
                actions = default_joint_vel[env_ids].clone()
            else:
                actions = default_joint_pos[env_ids].clone()
            actions = actions[None].repeat_interleave(self.delayed_steps, dim=0)
            self.delayed_joint_actions_buffer[:, env_ids] = actions

        # bookkeeping
        self.trial_counter[env_ids] += 1
        self.frame_id[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.last_rew[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def set_max_joint_velocities(self, scale):
        qdlim = torch.rad2deg(self.joint_vel_limits)

        scale = torch.atleast_1d(torch.as_tensor(scale)).to(self.device)
        if scale.size() != qdlim.size():
            scale = scale.expand_as(qdlim)

        self.joint_vel_scale = scale

        # Override velocity limits in simulation
        joints = self._get_all_joints(get_prim_at_path(self.robot.prim_path))
        for idx in self.active_joint_dofs:
            joints[idx].GetAttribute('physxJoint:maxJointVelocity').Set(
                qdlim[0, idx].item() * scale[0, idx].item()
            )
        return

    def _get_all_joints(self, prim):
        joints = []
        for child in prim.GetAllChildren():
            type = child.GetTypeName()
            if 'Joint' in type:
                if 'Fixed' not in type:
                    joints.append(child)
            else:
                joints.extend(self._get_all_joints(child))
        return joints

    def set_control_mode(self, mode: str):
        if not mode in ['velocity', 'position']:
            raise NotImplementedError
        self.control_mode = mode
        self._robots.switch_control_mode(mode)
        if mode == 'position':
            self.set_gains(self.kps, self.kds)
        elif mode == 'velocity':
            self.set_gains(self.kps*0, self.kds*2)

    def set_gains(self, kps, kds):
        self._robots.set_gains(
            kps, kds, joint_indices=self.active_joint_dofs, save_to_usd=False
        )

    def set_target_mode(self, mode: str):
        if not mode in ['velocity', 'position']:
            raise NotImplementedError
        self.target_mode = mode

    def set_low_level_controller(self, controller):
        self.low_level_controller = controller
        self.low_level_controller.set_task(self, override_params=True)

    def step_low_level_controller(self, count):
        if self.low_level_controller is not None:
            self.low_level_controller.step_controller(count)
        else:
            pass

    def get_robot_states(self):
        pos = self._robots.get_joint_positions(joint_indices=self.active_joint_dofs)
        vel = self._robots.get_joint_velocities(joint_indices=self.active_joint_dofs)
        return pos, vel

    def apply_control_target(self, control_target):
        if self.control_mode == 'position':
            control_target = torch.clamp(
                control_target,
                self.joint_limits[:, self.active_joint_dofs, 0],
                self.joint_limits[:, self.active_joint_dofs, 1]
            )
            self._robots.set_joint_position_targets(
                control_target, joint_indices=self.active_joint_dofs
            )
        elif self.control_mode == 'velocity':
            control_target = torch.clamp(
                control_target,
                -self.joint_vel_limits[:, self.active_joint_dofs],
                self.joint_vel_limits[:, self.active_joint_dofs]
            )
            self._robots.set_joint_velocity_targets(
                control_target, joint_indices=self.active_joint_dofs
            )

    def get_delayed_joint_actions(self, actions):
        if self.delay_actions:
            delayed_actions = self.delayed_joint_actions_buffer[-1]
            self.delayed_joint_actions_buffer = torch.cat(
                (actions.clone()[None], self.delayed_joint_actions_buffer[:-1]), dim=0
            )
        else:
            delayed_actions = actions
        return delayed_actions

    def pre_physics_step(self, actions):
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.actions = actions.clone().to(self.device)

        if self.target_mode == 'velocity':
            vel_limits = (self.joint_vel_limits * self.joint_vel_scale)[:, self.active_joint_dofs]
            joint_targets = torch.clamp(actions * vel_limits, -vel_limits, vel_limits)
        else:
            joint_range = self.joint_range[:, self.active_joint_dofs]
            joint_offset =  self.joint_offset[:, self.active_joint_dofs]
            joint_targets = torch.clamp(
                actions * joint_range / 2 + joint_offset,
                self.joint_limits[:, self.active_joint_dofs, 0],
                self.joint_limits[:, self.active_joint_dofs, 1]
            )

        joint_targets = self.get_delayed_joint_actions(joint_targets)
        self.joint_targets = joint_targets.clone()

        if self.low_level_controller is not None:
            self.low_level_controller.set_target(joint_targets)
        else:
            self.apply_control_target(joint_targets)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def post_reset(self) -> None:
        if not hasattr(self, 'active_joint_dofs'):
            self.active_joint_dofs = list(range(self.robot.joint_dof))
        if self.capture and not hasattr(self, 'rgb_annotators'):
            self.get_camera_sensor()

        self.kps, self.kds = self._robots.get_gains(joint_indices=self.active_joint_dofs)
        self.set_control_mode('position')

        limits = self.robot.qlim.to(device=self._device).T[None]
        self.joint_limits = limits
        self.joint_range = limits[..., 1] - limits[..., 0]
        self.joint_offset = (limits[..., 1] - limits[..., 0]) / 2 + limits[..., 0]

        self.joint_vel_limits = self.robot.qdlim[None]
        self.set_max_joint_velocities(self.max_vel_scale)

        qr = self.robot.default_joint_pos.clone().unsqueeze(0)
        self._default_joint_pos = qr.repeat_interleave(self.num_envs, 0)
        self._default_joint_vel = torch.zeros_like(self._default_joint_pos)
        self.joint_targets = self._default_joint_pos.clone()

        if self.delay_actions:
            self.delayed_joint_actions_buffer = torch.zeros(
                (self.delayed_steps, self.num_envs, self.num_joint_dof), device=self._device
            )

        self.extras["Successes"] = torch.zeros(self._num_envs,
                                               device=self._device,
                                               dtype=torch.long)
        self.extras["Failures"] = torch.zeros(self._num_envs,
                                               device=self._device,
                                                  dtype=torch.long)
        self.extras["Timeouts"] = torch.zeros(self._num_envs,
                                              device=self._device,
                                              dtype=torch.long)

    @abstractmethod
    def _calculate_metrics(self, **kwargs):
        pass

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0
        self.reset_buf = self.progress_buf >= self._max_episode_length

        successes, failures, timeouts = self._calculate_metrics()

        rew_diff = self.rew_buf - self.last_rew
        self.last_rew = self.rew_buf.clone()
        self.rew_buf = rew_diff

        # For evaluation of results
        self.extras["Successes"][:] = 0
        self.extras["Successes"][successes] = 1
        self.extras["Failures"][:] = 0
        self.extras["Failures"][failures] = 1
        self.extras["Timeouts"][:] = 0
        self.extras["Timeouts"][timeouts] = 1

        # For tensorboard plots during training
        if self.reset_buf.sum() > 0:
            self.extras["Success Rate"] = successes.sum() / self.reset_buf.sum()
            self.extras["Failure Rate"] = failures.sum() / self.reset_buf.sum()
            self.extras["Timeout Rate"] = timeouts.sum() / self.reset_buf.sum()
        else:
            self.extras["Success Rate"] = 0.0
            self.extras["Failure Rate"] = 0.0
            self.extras["Timeout Rate"] = 0.0

        self.old_actions = self.actions

    def is_done(self) -> None:
        pass  # Will be included in calculate metrics for efficiency

    @abstractmethod
    def get_camera_sensor(self) -> None:
        pass

    def capture_image(self):
        for i, rgb_annotator in enumerate(self.rgb_annotators):
            data = rgb_annotator.get_data()
            if len(data) > 0:  # Catch cases after reset
                frame_path = os.path.join(self.frame_path, f"Trial_{self.trial_counter[i].item():02}")
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
