from utilities.functions.time import perf_sleep

import torch
import time


class PhysicalEnv:

    def __init__(self, device):
        self.device = device
        self.clk = 0

    def set_task(self, task, **kwargs):
        self._task = task
        self._num_envs = self._task.num_envs

        self.observation_space = self._task.observation_space
        self.action_space = self._task.action_space
        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

        if hasattr(self.task, "has_gripper"):
            self.has_gripper = self.task.has_gripper
        if hasattr(self.task, "joint_observation_space"):
            self.joint_observation_space = self.task.joint_observation_space
        if hasattr(self.task, "task_observation_space"):
            self.task_observation_space = self.task.task_observation_space
        if hasattr(self.task, "joint_space"):
            self.joint_space = self.task.joint_space

    @property
    def task(self):
        return self._task

    @property
    def step_time(self):
        return self._task.step_time

    def shutdown(self):
        self._task.shutdown()

    def step(self, action):
        done = self._task.step(action)
        print("Step Time: {:.2f} ms".format((time.perf_counter() - self.clk) * 1000))

        # Ensure system maintains step time
        diff_to_step_time = self.step_time - (time.perf_counter() - self.clk)
        if diff_to_step_time > 1e-6:
            perf_sleep(diff_to_step_time)

        print("Perf. Step Time: {:.2f} ms\n".format((time.perf_counter() - self.clk) * 1000))

        self.clk = time.perf_counter()
        obs, times = self.get_observations()
        for k, v in times.items():
            print(k, ": {:.2f} ms".format(v * 1000))
        return obs, done

    def reset(self):
        self._task.reset()
        self.clk = time.perf_counter()
        obs, times = self.get_observations()
        for k, v in times.items():
            print(k, ": {:.2f} ms".format(v * 1000))
        return obs

    def get_observations(self):
        return self._task.get_observations()
