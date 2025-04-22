import os
import torch
import numpy as np
from typing import Optional
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema


class Foosball(Robot):
    joint_dof = 16

    def __init__(
            self,
            prim_path: str,
            name: Optional[str] = "foosball",
            usd_path: Optional[str] = None,
            translation: Optional[torch.tensor] = None,
            orientation: Optional[torch.tensor] = None,
            device: str = 'cpu'
    ) -> None:
        """[summary]
        """
        # TODO: Add scaling for table

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            self._usd_path = os.path.join(
                root_dir, "usd/Foosball_Instanceable.usd"
            )

        self.reference = add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        self.qlim = torch.tensor([
            [-0.123, -0.179, -0.0615, -0.1185] * 2 + [-2*np.pi] * 8,
            [ 0.123,  0.179,  0.0615,  0.1185] * 2 + [ 2*np.pi] * 8
        ], device=device)

        t = 0.01108 * np.pi  # Belt drive transmission factor (r*pi)
        self.qdlim = torch.tensor([100*t] * 8 + [100*np.pi] * 8, device=device)
        self.qddlim = torch.tensor([3000*t] * 8 + [3000*np.pi] * 8, device=device)
        self.qdddlim = torch.tensor([11_667*t] * 8 + [11_667*np.pi] * 8, device=device)
        # self.qdddlim = torch.tensor([11_667 * t] * 8 + [133_333 * np.pi] * 8, device=device)
        # Jerk matches real system settings (low due to vibrations)

        self.default_joint_pos = torch.zeros_like(self.qdlim)

        # TODO: Calc in relation to table size and rescale if necessary
        self.figure_positions = {
            "Keeper_W": torch.tensor([[0.522], [0]], device=device),
            "Defense_W": torch.tensor([[0.37295, 0.37295], [-0.1235, 0.1235]], device=device),
            "Mid_W": torch.tensor([[0.07469, 0.07469, 0.07469, 0.07469, 0.07469], [-0.241, -0.1205, 0, 0.1205, 0.241]], device=device),
            "Offense_W": torch.tensor([[-0.22372, -0.22372, -0.22372], [-0.184, 0, 0.184]], device=device),
            #
            "Keeper_B": torch.tensor([[-0.522], [0]], device=device),
            "Defense_B": torch.tensor([[-0.37295, -0.37295], [-0.1235, 0.1235]], device=device),
            "Mid_B": torch.tensor([[-0.07469, -0.07469, -0.07469, -0.07469, -0.07469], [-0.241, -0.1205, 0, 0.1205, 0.241]], device=device),
            "Offense_B": torch.tensor([[0.22372, 0.22372, 0.22372], [-0.184, 0, 0.184]], device=device)
        }

        self.dof_paths = [
            "Keeper_W_PrismaticJoint",
            "Defense_W_PrismaticJoint",
            "Mid_W_PrismaticJoint",
            "Offense_W_PrismaticJoint",
            "Keeper_B_PrismaticJoint",
            "Defense_B_PrismaticJoint",
            "Mid_B_PrismaticJoint",
            "Offense_B_PrismaticJoint",
            "Keeper_W_RevoluteJoint",
            "Defense_W_RevoluteJoint",
            "Mid_W_RevoluteJoint",
            "Offense_W_RevoluteJoint",
            "Keeper_B_RevoluteJoint",
            "Defense_B_RevoluteJoint",
            "Mid_B_RevoluteJoint",
            "Offense_B_RevoluteJoint"
        ]

        self.dof_paths_W = [
            "Keeper_W_PrismaticJoint",
            "Defense_W_PrismaticJoint",
            "Mid_W_PrismaticJoint",
            "Offense_W_PrismaticJoint",
            "Keeper_W_RevoluteJoint",
            "Defense_W_RevoluteJoint",
            "Mid_W_RevoluteJoint",
            "Offense_W_RevoluteJoint",
        ]

        self.dof_paths_B = [
            "Keeper_B_PrismaticJoint",
            "Defense_B_PrismaticJoint",
            "Mid_B_PrismaticJoint",
            "Offense_B_PrismaticJoint",
            "Keeper_B_RevoluteJoint",
            "Defense_B_RevoluteJoint",
            "Mid_B_RevoluteJoint",
            "Offense_B_RevoluteJoint"
        ]

        self.dof_paths_rev = [
            "Keeper_W_RevoluteJoint",
            "Defense_W_RevoluteJoint",
            "Mid_W_RevoluteJoint",
            "Offense_W_RevoluteJoint",
            "Keeper_B_RevoluteJoint",
            "Defense_B_RevoluteJoint",
            "Mid_B_RevoluteJoint",
            "Offense_B_RevoluteJoint"
        ]

        self.dof_paths_pris = [
            "Keeper_W_PrismaticJoint",
            "Defense_W_PrismaticJoint",
            "Mid_W_PrismaticJoint",
            "Offense_W_PrismaticJoint",
            "Keeper_B_PrismaticJoint",
            "Defense_B_PrismaticJoint",
            "Mid_B_PrismaticJoint",
            "Offense_B_PrismaticJoint"
        ]

        self.rod_paths_W = [
            "White/Keeper_W",
            "White/Defense_W",
            "White/Mid_W",
            "White/Offense_W"
        ]

        self.rod_paths_B = [
            "Black/Keeper_B",
            "Black/Defense_B",
            "Black/Mid_B",
            "Black/Offense_B"
        ]
