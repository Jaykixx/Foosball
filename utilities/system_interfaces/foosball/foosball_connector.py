from utilities.system_interfaces.base_connector import BaseConnector
from utilities.system_interfaces.foosball.motor_can402 import MotorCAN402
from threading import Thread
import canopen
import torch
import time


class FoosballConnector(BaseConnector):

    def __init__(self, sys_cfg, device='cuda:0'):
        BaseConnector.__init__(self, sys_cfg, device)

        joint_infos = self.sys_cfg['system']['drives']
        self._joint_info = {k: joint_infos[k] for k in self.settings['active_drives']}
        self._joint_order = 0
        self._joints = None

    def establish_connection(self):
        import pathlib
        import os
        dir_path = pathlib.Path(__file__).parent.resolve()

        self.network = canopen.Network()
        self.network.connect(bustype='pcan', channel='PCAN_USBBUS1', bitrate=1000000)
        time.sleep(0.1)

        self._joints = []
        for info in self._joint_info.values():
            eds_path = os.path.join(dir_path, 'eds', info["eds_location"])
            node = canopen.RemoteNode(info["node_id"], eds_path)
            self.network.add_node(node)
            node.nmt.state = "PRE-OPERATIONAL"
            joint = MotorCAN402(
                node,
                rename_map=info["rename_map"],
                motion_profiles=info["motion_profiles_cfg"],
                reversed=info["reversed"]
            )
            joint.set_name(info["name"])
            joint.set_command_limits(*info["command_limits"])
            joint.set_dof_limits(*info["range_of_motion"])
            self._joints.append(joint)

        self.start()

    def get_joint_observations(self):
        pos, vel = [], []
        for joint in self._joints:
            pos.append(joint.get_current_dof_pos())
            vel.append(joint.get_current_dof_vel())
        obs = torch.cat(
            (torch.tensor(pos, dtype=torch.float32, device=self.device),
             torch.tensor(vel, dtype=torch.float32, device=self.device)),
            dim=-1
        )
        return obs

    def get_joint_positions(self):
        pos = []
        for joint in self._joints:
            pos.append(joint.get_current_dof_pos())
        return torch.tensor(pos, dtype=torch.float32, device=self.device)

    def get_joint_velocities(self):
        vel = []
        for joint in self._joints:
            vel.append(joint.get_current_dof_vel())
        return torch.tensor(vel, dtype=torch.float32, device=self.device)

    def shutdown(self):
        self.stop()
        [joint.shutdown() for joint in self._joints]
        self.network.disconnect()
        time.sleep(0.5)

    def apply_actions(self, actions):
        [self._joints[i].set_target(v.item()) for i, v in enumerate(actions)]

    def start(self):
        self.threads = []
        for joint in self._joints:
            joint.to_switch_on_disabled()
            joint.operating_mode = "pp"
            joint.to_operational()
            thread = Thread(target=joint.run_pos_tracking)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        if hasattr(self, 'threads') and len(self.threads) > 0:
            [joint.set_target(None) for joint in self._joints]
            [thread.join() for thread in self.threads]
