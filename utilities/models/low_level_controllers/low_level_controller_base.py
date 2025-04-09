from abc import abstractmethod
import torch


class LowLevelControllerBase:

    def __init__(self, control_mode: str, target_mode: str):
        self.control_mode = control_mode
        self.target_mode = target_mode
        self._task = None

        self.dt = None
        self.max_steps = None

    def set_task(self, task, override_params=False):
        self._task = task
        self._task.set_control_mode(self.control_mode)
        self._task.set_target_mode(self.target_mode)
        if override_params:
            self.max_steps = self._task.control_frequency_inv
            self.dt = self._task.phys_dt

    def apply_control_target(self, control_target):
        self._task.apply_control_target(control_target)

    def get_robot_states(self):
        return self._task.get_robot_states()

    @abstractmethod
    def step_controller(self, count):
        pass

    @abstractmethod
    def set_target(self, target):
        pass
