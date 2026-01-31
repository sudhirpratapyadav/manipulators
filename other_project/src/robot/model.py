"""
Pinocchio-based robot model for Kinova Gen3.
Provides gravity compensation, forward kinematics, and differential IK.
"""

import os
import numpy as np
import pinocchio as pin
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "..", "assets", "robots", "kinova", "urdf", "gen3_2f85.urdf")

JOINT_NAMES = [
    'gen3_joint_1', 'gen3_joint_2', 'gen3_joint_3', 'gen3_joint_4',
    'gen3_joint_5', 'gen3_joint_6', 'gen3_joint_7'
]
EE_FRAME = "gen3_end_effector_link"


class RobotModel:
    """Combined gravity compensation and differential IK using Pinocchio."""

    def __init__(self, urdf_path: str = None):
        if urdf_path is None:
            urdf_path = URDF_PATH

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Get EE frame ID
        self.ee_frame_id = self.model.getFrameId(EE_FRAME)
        if self.ee_frame_id == self.model.nframes:
            raise ValueError(f"Frame '{EE_FRAME}' not found")

        # Precompute joint indices
        self.v_indices = []
        self.q_info = []
        for name in JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(f"Joint '{name}' not found")
            joint = self.model.joints[jid]
            self.v_indices.append(joint.idx_v)
            self.q_info.append((joint.idx_q, joint.nq))

    def _to_pinocchio_config(self, q_arm: np.ndarray) -> np.ndarray:
        """Convert 7-DOF arm angles to full Pinocchio configuration."""
        q_full = pin.neutral(self.model)
        for i, (idx_q, nq) in enumerate(self.q_info):
            if nq == 1:
                q_full[idx_q] = q_arm[i]
            else:  # nq == 2: unbounded joint (cos, sin)
                q_full[idx_q] = np.cos(q_arm[i])
                q_full[idx_q + 1] = np.sin(q_arm[i])
        return q_full

    def gravity(self, q_rad: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torques.

        Args:
            q_rad: Joint angles (7,) in radians

        Returns:
            tau_g: Gravity torques (7,) in Nm
        """
        q_full = self._to_pinocchio_config(q_rad)
        pin.computeGeneralizedGravity(self.model, self.data, q_full)
        return np.array([self.data.g[v_idx] for v_idx in self.v_indices])

    def forward_kinematics(self, q_rad: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.

        Args:
            q_rad: Joint angles (7,) in radians

        Returns:
            pose: 6D array [x, y, z, rx, ry, rz] (axis-angle)
        """
        q_full = self._to_pinocchio_config(q_rad)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        pose_se3 = self.data.oMf[self.ee_frame_id]
        position = pose_se3.translation.copy()
        rotation_aa = pin.log3(pose_se3.rotation)
        return np.concatenate([position, rotation_aa])

    def diff_ik_step(
        self,
        q_rad: np.ndarray,
        target_pose: np.ndarray,
        max_step: float = 0.1,
        lambda_min: float = 0.01,
        lambda_max: float = 0.5,
        manip_threshold: float = 0.01,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Single differential IK step with damped least squares.

        Args:
            q_rad: Current joint angles (7,) in radians
            target_pose: Target EE pose [x, y, z, rx, ry, rz]
            max_step: Maximum joint angle change (radians)
            lambda_min: Minimum damping factor
            lambda_max: Maximum damping factor
            manip_threshold: Manipulability threshold for adaptive damping

        Returns:
            q_new: New joint angles (7,) in radians
            pos_error_norm: Position error magnitude (meters)
            rot_error_norm: Rotation error magnitude (radians)
        """
        q_full = self._to_pinocchio_config(q_rad)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        # Get current pose and compute error
        current_pose = self.data.oMf[self.ee_frame_id]
        pos_error, rot_error = self._compute_pose_error(current_pose, target_pose)

        pos_error_norm = np.linalg.norm(pos_error)
        rot_error_norm = np.linalg.norm(rot_error)

        # Compute Jacobian
        J = self._compute_jacobian(q_full)
        JT = J.T
        JJT = J @ JT

        # Adaptive damping based on manipulability
        manip = np.sqrt(max(np.linalg.det(JJT), 0))
        if manip < manip_threshold:
            lambda_dls = lambda_max * (1 - manip / manip_threshold)
        else:
            lambda_dls = lambda_min

        # Damped least squares
        delta_pose = np.concatenate([pos_error, rot_error])
        lambda_sq_I = (lambda_dls ** 2) * np.eye(6)
        delta_q = JT @ np.linalg.solve(JJT + lambda_sq_I, delta_pose)

        # Limit step size
        step_norm = np.linalg.norm(delta_q)
        if step_norm > max_step:
            delta_q = delta_q * (max_step / step_norm)

        q_new = np.array(q_rad) + delta_q
        return q_new, pos_error_norm, rot_error_norm

    def _compute_jacobian(self, q_full: np.ndarray) -> np.ndarray:
        """Compute 6x7 Jacobian for arm joints."""
        J_full = pin.computeFrameJacobian(
            self.model, self.data, q_full,
            self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        return J_full[:, self.v_indices]

    def _compute_pose_error(
        self,
        current_pose_se3,
        target_pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute position and rotation error."""
        # Position error
        pos_error = target_pose[:3] - current_pose_se3.translation

        # Rotation error
        current_rot_aa = pin.log3(current_pose_se3.rotation)
        R_desired = pin.exp3(target_pose[3:])
        R_current = pin.exp3(current_rot_aa)
        R_error = R_desired @ R_current.T
        rot_error = pin.log3(R_error)

        return pos_error, rot_error
