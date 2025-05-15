from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUEnv
import torch


class CustomVecEnvRLGames(VecEnvRLGames):
    """ Wraps the default Isaac Sim RL Environment to allow for customization
    and higher flexibility, e.g. access to private properties. """

    def __init__(
            self, headless: bool,
            sim_device: int = 0,
            enable_livestream: bool = False,
            enable_viewport: bool = False,
            launch_simulation_app: bool = True,
            experience: str = None
    ) -> None:
        super(CustomVecEnvRLGames, self).__init__(
            headless, sim_device, enable_livestream, enable_viewport, launch_simulation_app, experience
        )
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.isaac")

        # enable multi agent settings
        self.full_actions = lambda actions: actions

    def set_task(self, **kwargs) -> None:
        super().set_task(**kwargs)
        # Override phys_dt in task since world rounds to full Hz Frequency
        self._task.phys_dt = self._world.get_physics_dt()
        if hasattr(self.task, "num_opponents"):
            self.full_actions = lambda action: self._task.full_actions(action)

        if hasattr(self.task, "has_gripper"):
            self.has_gripper = self.task.has_gripper
        if hasattr(self.task, "joint_observation_space"):
            self.joint_observation_space = self.task.joint_observation_space
        if hasattr(self.task, "task_observation_space"):
            self.task_observation_space = self.task.task_observation_space
        if hasattr(self.task, "joint_space"):
            self.joint_space = self.task.joint_space
        if hasattr(self.task, "_num_obj_types"):
            self.num_object_types = self.task._num_obj_types

    @property
    def task(self):
        return self._task

    @property
    def progress_buffer(self):
        return self._task.progress_buf

    def get_number_of_agents(self):
        return self._task.num_agents

    def step(self, actions):
        actions = self.full_actions(actions)

        # only enable rendering when we are recording, or if the task already has it enabled
        to_render = self._render
        if self._record:
            if not hasattr(self, "step_count"):
                self.step_count = 0
            if self.step_count % self._task.cfg["recording_interval"] == 0:
                self.is_recording = True
                self.record_length = 0
            if self.is_recording:
                self.record_length += 1
                if self.record_length > self._task.cfg["recording_length"]:
                    self.is_recording = False
            if self.is_recording:
                to_render = True
            else:
                if (self._task.cfg["headless"] and not self._task.enable_cameras and not self._task.cfg["enable_livestream"]):
                    to_render = False
            self.step_count += 1

        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(
                actions=actions, reset_buf=self._task.reset_buf
            )

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device)

        self._task.pre_physics_step(actions)

        if (self.sim_frame_count + self._task.control_frequency_inv) % self._task.rendering_interval == 0:
            for count in range(self._task.control_frequency_inv - 1):
                self._task.step_low_level_controller(count)
                self._world.step(render=False)
                self.sim_frame_count += 1
            self._task.step_low_level_controller(self._task.control_frequency_inv - 1)
            self._world.step(render=to_render)
            self.sim_frame_count += 1
        else:
            for count in range(self._task.control_frequency_inv):
                self._task.step_trajectory(count)
                self._world.step(render=False)
                self.sim_frame_count += 1

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device), reset_buf=self._task.reset_buf
            )

        self._states = self._task.get_states()
        self._process_data()

        obs_dict = {"obs": self._obs, "states": self._states}

        return obs_dict, self._rew, self._resets, self._extras

    def reset(self):
        obs = super().reset()
        if self._task.num_agents > 1:
            self._task.prepare_opponent()
            self._task.obs_bufs = obs
        return obs

    def set_low_level_controller(self, controller):
        self._task.set_low_level_controller(controller)

    def set_weights(self, indices, weights):
        self._task.update_weights(indices, weights)


class SelfPlayRLGPUEnv(RLGPUEnv):
    def set_weights(self, indices, weights) -> None:
        self.env.set_weights(indices, weights)
