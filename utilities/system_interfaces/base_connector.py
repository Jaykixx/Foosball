from signal import signal, SIGINT
from abc import abstractmethod


class BaseConnector:

    def __init__(self, sys_cfg, device='cuda:0'):
        self.device = device
        self.sys_cfg = sys_cfg
        self.settings = self.sys_cfg['system']['settings']

        self._step_time = 1 / self.settings['control_frequency']
        self._joint_info = None

        signal(SIGINT, lambda h, f: (self.shutdown(), exit(0)))

    @property
    def step_time(self):
        return self._step_time

    @property
    def joint_info(self):
        return self._joint_info

    @abstractmethod
    def establish_connection(self):
        pass

    @abstractmethod
    def get_joint_observations(self):
        pass

    @abstractmethod
    def get_joint_positions(self):
        pass

    @abstractmethod
    def get_joint_velocities(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def apply_actions(self, actions):
        pass
