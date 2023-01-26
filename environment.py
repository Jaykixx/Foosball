from omniisaacgymenvs.tasks.utils.usd_utils import create_distant_light
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects.sphere import DynamicSphere
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.tasks import BaseTask
from omni.isaac.cloner import GridCloner
from gym import spaces
import numpy as np
import omni.kit
import torch
import os


class FoosballTask(BaseTask):

    def __init__(self, name, env, offset=None) -> None:
        super().__init__(name=name, offset=offset)

        self._device = 'cuda:0'
        self._env = env
        self._num_agents = 1
        self._env_spacing = 4.0
        self._num_envs = 4

        # initialize data spaces (defaults to gym.Box)
        if not hasattr(self, "action_space"):
            self.action_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        if not hasattr(self, "observation_space"):
            self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self.cleanup()

    def cleanup(self) -> None:
        """ Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = torch.zeros((self._num_envs, self.num_observations), device=self._device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.extras = {}

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)

        collision_filter_global_paths = list()

        # Add foosball table
        root_dir = os.path.dirname(os.path.abspath(__file__))
        usd_path = os.path.join(root_dir, "foosball_python.usd")
        foosball = add_reference_to_stage(usd_path, self.default_zero_env_path)

        # Add game ball
        ball_path = self.default_zero_env_path + "/Ball"
        self.ball_start_position = torch.tensor([-0.48, 0, 0.79025])
        DynamicSphere(ball_path, position=self.ball_start_position, radius=0.01725, mass=0.023)

        # Ground Plane
        self._ground_plane_path = "/World/defaultGroundPlane"
        collision_filter_global_paths.append(self._ground_plane_path)
        scene.add_default_ground_plane(prim_path=self._ground_plane_path)

        # Clone environment
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths)
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        # self._cloner.filter_collisions(
        #     self._env._world.get_physics_context().prim_path, "/World/collisions", prim_paths, collision_filter_global_paths)

        # Get robot articulations
        self._robots = ArticulationView(prim_paths_expr="/World/envs/env_.*/Foosball", name="robot_view")
        scene.add(self._robots)

        # Get ball view
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/env_.*/Ball", name="ball_view")

        # Set up visuals
        self.set_initial_camera_params(camera_position=[10, 10, 3], camera_target=[0, 0, 0])
        create_distant_light()

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position("/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True)
        viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

    @property
    def default_base_env_path(self):
        """ Retrieves default path to the parent of all env prims.
        Returns:
            default_base_env_path(str): Defaults to "/World/envs".
        """
        return "/World/envs"

    @property
    def default_zero_env_path(self):
        """ Retrieves default path to the first env prim (index 0).
        Returns:
            default_zero_env_path(str): Defaults to "/World/envs/env_0".
        """
        return f"{self.default_base_env_path}/env_0"

    @property
    def num_envs(self):
        """ Retrieves number of environments for task.
        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    @property
    def num_actions(self):
        """ Retrieves dimension of actions.
        Returns:
            num_actions(int): Dimension of actions.
        """
        return 4

    @property
    def num_observations(self):
        """ Retrieves dimension of observations.
        Returns:
            num_observations(int): Dimension of observations.
        """
        return 8

    @property
    def num_agents(self):
        """ Retrieves number of agents for multi-agent environments.
        Returns:
            num_agents(int): Dimension of states.
        """
        return self._num_agents

    def get_extras(self):
        """ API for retrieving extras data for RL.
        Returns:
            extras(dict): Dictionary containing extras data.
        """
        return self.extras

    def reset(self):
        """ Flags all environments for reset.
        """
        self.reset_buf = torch.ones_like(self.reset_buf)
        return self.get_observations()

    def pre_physics_step(self, actions):
        """ Optionally implemented by individual task classes to process actions.
        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        self._robots.set_joint_position_targets(actions)
        # pass

    def post_physics_step(self):
        """ Processes RL required computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.
        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            self.get_observations()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self) -> dict:
        self.obs_buf = self._robots.get_joint_positions()
        return {self._name: {"obs_buf": self.obs_buf}}

    def is_done(self) -> bool:
        return False

    def calculate_metrics(self) -> None:
        pass