from utilities.system_interfaces.base_connector import BaseConnector
from abc import abstractmethod
from gym import spaces
import numpy as np
import torch
import time


class BaseInterface:

    def __init__(self, **kwargs):
        self.sys_connector = self.get_system_connector()
        self.sys_connector.establish_connection()

        # initialize variables
        self._num_envs = 1
        if not hasattr(self, "_num_joint_observations"):
            self._num_joint_observations = 2 * self.num_actions
        if not hasattr(self, "_num_task_observations"):
            self._num_task_observations = 0
        if not hasattr(self, "_num_observations"):
            self._num_observations = self._num_task_observations + self._num_joint_observations
        if not hasattr(self, "_num_states"):
            self._num_states = 1
        if not hasattr(self, "_num_agents"):
            self._num_agents = 1
        self.progress_buf = None

        # initialize data spaces (defaults to gym.Box)
        if not hasattr(self, "action_space"):
            self.action_space = spaces.Box(
                np.ones(self.num_actions, dtype=np.float32) * -1.0, np.ones(self.num_actions, dtype=np.float32) * 1.0
            )
        if not hasattr(self, "observation_space"):
            self.observation_space = spaces.Box(
                np.ones(self.num_observations, dtype=np.float32) * -np.Inf,
                np.ones(self.num_observations, dtype=np.float32) * np.Inf,
            )
        if not hasattr(self, "state_space"):
            self.state_space = spaces.Box(
                np.ones(self.num_states, dtype=np.float32) * -np.Inf,
                np.ones(self.num_states, dtype=np.float32) * np.Inf,
            )
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

        self.clk = time.perf_counter()

    @property
    def step_time(self):
        return self.sys_connector.step_time

    @abstractmethod
    def get_system_connector(self) -> BaseConnector:
        pass

    def shutdown(self):
        self.sys_connector.shutdown()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def get_task_observation(self):
        pass

    def get_observations(self):
        clk = time.perf_counter()
        task_obs, times = self.get_task_observation()

        joint_obs_clk = time.perf_counter()
        joint_obs = self.sys_connector.get_joint_observations()[None]

        times['Joint Obs Time'] = time.perf_counter() - joint_obs_clk
        times['Full Obs Time'] = time.perf_counter() - clk

        if task_obs is not None:
            return torch.cat((joint_obs, task_obs), dim=-1), times
        else:
            return joint_obs, times

    @property
    def num_actions(self):
        """Retrieves dimension of actions.

        Returns:
            num_actions(int): Dimension of actions.
        """
        return self._num_actions

    @property
    def num_envs(self):
        """Retrieves number of environments for task.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    @property
    def num_observations(self):
        """Retrieves dimension of observations.

        Returns:
            num_observations(int): Dimension of observations.
        """
        return self._num_observations

    @property
    def num_joint_observations(self):
        """Retrieves dimension of observations.

        Returns:
            num_observations(int): Dimension of observations.
        """
        return self._num_joint_observations

    @property
    def num_task_observations(self):
        """Retrieves dimension of observations.

        Returns:
            num_observations(int): Dimension of observations.
        """
        return self._num_task_observations

    @property
    def num_states(self):
        """Retrieves dimesion of states.

        Returns:
            num_states(int): Dimension of states.
        """
        return self._num_states

    @property
    def num_agents(self):
        """Retrieves number of agents for multi-agent environments.

        Returns:
            num_agents(int): Dimension of states.
        """
        return self._num_agents