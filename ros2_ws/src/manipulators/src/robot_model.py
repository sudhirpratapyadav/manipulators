"""
Pinocchio robot model wrapper.

Loads URDF, builds a reduced model (arm joints only, gripper locked),
and provides gravity, Jacobian, and FK computations.
"""

import numpy as np
import pinocchio as pin


ARM_JOINT_NAMES = [f"gen3_joint_{i}" for i in range(1, 8)]


class RobotModel:
    """Pinocchio model for the Kinova Gen3 arm (7-DOF, gripper joints locked)."""

    def __init__(self, urdf_path: str, ee_frame: str = "gen3_end_effector_link"):
        # Load full model from URDF
        full_model = pin.buildModelFromUrdf(urdf_path)

        # Identify arm joint IDs
        arm_ids = set()
        for name in ARM_JOINT_NAMES:
            jid = full_model.getJointId(name)
            if jid >= full_model.njoints:
                raise ValueError(f"Joint '{name}' not found in URDF")
            arm_ids.add(jid)

        # Lock every non-arm joint (gripper joints) at neutral
        joints_to_lock = [
            i for i in range(1, full_model.njoints) if i not in arm_ids
        ]
        q_ref = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, joints_to_lock, q_ref)
        self.data = self.model.createData()

        # Cache EE frame ID
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        if self.ee_frame_id >= self.model.nframes:
            raise ValueError(f"Frame '{ee_frame}' not found in model")

        self.nq = self.model.nq  # should be 7

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Compute generalized gravity torques (7,)."""
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute EE Jacobian (6x7) in LOCAL_WORLD_ALIGNED frame.

        Rows: [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z].
        """
        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )

    def fk(self, q: np.ndarray):
        """
        Forward kinematics for the EE frame.

        Returns:
            position: (3,) translation
            rotation: (3,3) rotation matrix
        """
        pin.framesForwardKinematics(self.model, self.data, q)
        oMf = self.data.oMf[self.ee_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()
