from motor402.motor import RPDOConfig, TPDOConfig
from motor402.motor import Motor as BaseMotor

from motor402.utility import *
from threading import Lock
from queue import Queue
import canopen
import time


rename_map = {
    "controlword": "Controlword",
    "statusword": "Statusword",
    "operating_mode": "Modes of operation",
    "target_position": "Target Position",
    "profile_velocity": "Profile velocity",
    "target_velocity": "Target velocity",
    "homing_method": "Homing Method 1",
    "position_actual_value": "Position actual value",
    "velocity_actual_value": "Velocity actual value",
    "switches": "Switch Parameters 1",
    "microstep_resolution": "Microstep Resolution 1"
}

motion_profiles_cfg = {
    "index": "operating_mode",
    "profiles": {
        "no_mode": 0,
        "pp": 1,
        "pv": 2,
        "hm": 6,
        "csp": 8,
        "csv": 9
    }
}


class MotorCAN402(BaseMotor):  # TODO: Look at PDO Task in base class

    def __init__(
            self,
            node: canopen.RemoteNode,
            rename_map: dict = rename_map,
            motion_profiles: dict = motion_profiles_cfg,
            *,
            reversed=False,
            cw_index=0x6040,
            sw_index=0x6041
    ):
        BaseMotor.__init__(
            self,
            node,
            rename_map,
            motion_profiles,
            controlword_index=cw_index,
            statusword_index=sw_index
        )

        self.name = None

        self.lock = Lock()
        self.actual_pos = self.get("position_actual_value")
        self.actual_vel = self.get("velocity_actual_value")
        self.targets = Queue(maxsize=1)
        # self.target = self.actual_pos

        self._min_command = self.actual_pos
        self._max_command = self.actual_pos
        self._command_half_range = (self._max_command - self._min_command) / 2
        self._command_offset = self._min_command + self._command_half_range

        self._min_dof_pos = self.actual_pos
        self._max_dof_pos = self.actual_pos
        self._dof_half_range = (self._max_dof_pos - self._min_dof_pos) / 2
        self._dof_offset = self._min_dof_pos + self._dof_half_range

        self.reversed = reversed

    def set_name(self, name):
        self.name = name

    def set_dof_limits(self, min, max):
        self._min_dof_pos = min
        self._max_dof_pos = max
        self._dof_half_range = (self._max_dof_pos - self._min_dof_pos) / 2
        self._dof_offset = self._min_dof_pos + self._dof_half_range

    def set_command_limits(self, min, max):
        self._min_command = min
        self._max_command = max
        self._command_half_range = (self._max_command - self._min_command) / 2
        self._command_offset = self._min_command + self._command_half_range

    def update_actual_pos(self):
        pos = self.get("position_actual_value")
        with self.lock:
            self.actual_pos = pos

    def update_actual_vel(self):
        vel = self.get("velocity_actual_value")
        with self.lock:
            self.actual_vel = vel

    def update_state(self):
        self.update_actual_pos()
        self.update_actual_vel()

    def get_current_dof_pos(self):
        with self.lock:
            actual_pos = self.actual_pos
        normed = (actual_pos - self._command_offset) / self._command_half_range
        dof_pos = normed * self._dof_half_range + self._dof_offset
        if self.reversed:
            dof_pos = - dof_pos
        return dof_pos

    def get_current_dof_vel(self):
        # TODO: Assumption - vel is in incrememnts per second
        # TODO: Check increments per rotation!
        with self.lock:
            actual_vel = self.actual_vel * 3585 / 60
        dof_vel = actual_vel * self._dof_half_range / self._command_half_range
        if self.reversed:
            dof_vel = - dof_vel
        return dof_vel

    def set_target(self, target):
        self.targets.put(target)

    def set_action_target(self, action):
        if self.reversed:
            action = -action
        target = self.compute_command_from_action(action)
        self.set_target(target)

    def compute_command_from_action(self, action):
        """ Assumes -1 to 1 inputs """
        command = action * self._command_half_range + self._command_offset
        command = min(command, self._max_command)
        command = max(command, self._min_command)
        return command

    def move_to_target(self, value, *, target_index='target_position', profile='pp', relative=False):
        # sets the targets position, but do not execute it until the operation is enabled
        self.set(target_index, int32(value))

        # Sets bits 0-5 to initiate movement
        self.set(self._cw_index, uint16(63))

        # Resets bit 4 to accept new move command in the future
        # Manual states it works without reset but experiment proved it doesn't
        self.set(self._cw_index, uint16(47))

    def run_pos_tracking(self):
        # TODO: Only works if targets will be emptied before next comes in
        # In other words execution time << observation time
        # Assuming 1ms per message (Controller clk time):
        while True:
            # loop runs at 500 Hz for empty queue
            if self.targets.empty():
                self.update_state()  # 2 ms
            else:
                # loop runs at 200 Hz for full queue
                target = self.targets.get()
                if target is None:
                    break
                else:
                    self.move_to_target(target)  # 3 ms
                    # Ensures that state is updated at least once per step
                    self.update_state()  # 2 ms
