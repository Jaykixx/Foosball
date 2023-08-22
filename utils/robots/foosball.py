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

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            # self._usd_path = os.path.join(
            #     root_dir, "../../environments/Foosball/Models/Foosball_v2_Fully_Assembled.usd"
            # )
            self._usd_path = os.path.join(
                root_dir, "../../environments/Foosball/Models/Foosball_v5.usd"
            )

        self.reference = add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        self.qdlim = torch.tensor([4.0] * 8 + [100*np.pi] * 8, device=device)

        self.figure_positions = {
            'Keeper': torch.tensor([0], device=device),
            'Defense': torch.tensor([-0.1235, 0.1235], device=device),
            'Mid': torch.tensor([-0.241, -0.1205, 0, 0.1205, 0.241], device=device),
            'Offense': torch.tensor([-0.184, 0, 0.184], device=device)
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

    def apply_joint_settings(self):
        drive_type = ["linear"] * 8 + ["angular"] * 8
        default_dof_pos = [0.0] * 8 + [70.0] * 8
        stiffness = [1000.0] * 8 + [400*np.pi/180] * 8
        damping = [50.0] * 8 + [80*np.pi/180] * 8
        max_force = [70.0] * 8 + [0.47] * 8
        max_velocity = [4.0] * 8 + [18000.0] * 8

        for i, dof in enumerate(self.dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(
                get_prim_at_path(f"{self.prim_path}/{dof}")
            ).CreateMaxJointVelocityAttr().Set(max_velocity[i])
