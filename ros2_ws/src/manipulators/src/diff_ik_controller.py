"""
Differential IK controller with gravity compensation.

Target EE pose  -->  diff-IK (damped pseudoinverse)  -->  desired joint velocities
                -->  joint-space damping + gravity comp  -->  joint torques
"""

import numpy as np

from .robot_model import RobotModel
from .utility import pose_error


class DiffIKController:
    """
    Computes joint torques to track a Cartesian pose target.

    Each cycle:
      1. FK to get current EE pose
      2. Compute 6D pose error (position + orientation)
      3. Desired EE twist = Kp_task * error
      4. Desired joint velocity = J_damped_pinv * twist
      5. q_desired = q + dq_desired * dt
      6. Torque = Kp_joint * (q_desired - q) + Kd_joint * (dq_desired - dq_actual) + gravity(q)
    """

    def __init__(
        self,
        model: RobotModel,
        kp_task: np.ndarray,
        kp_joint: np.ndarray,
        kd_joint: np.ndarray,
        dt: float,
        damping: float = 0.01,
        max_joint_velocity: float = 1.5,
        max_torque: np.ndarray = None,
    ):
        self.model = model
        self.kp_task = np.asarray(kp_task, dtype=float)       # (6,)
        self.kp_joint = np.asarray(kp_joint, dtype=float)     # (7,)
        self.kd_joint = np.asarray(kd_joint, dtype=float)     # (7,)
        self.dt = dt
        self.damping = damping
        self.max_dq = max_joint_velocity
        self.max_torque = (
            np.asarray(max_torque, dtype=float) if max_torque is not None
            else np.full(7, 50.0)
        )

    def compute(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint torques to track the target pose.

        Args:
            target_pos: desired EE position (3,)
            target_quat_xyzw: desired EE orientation [x,y,z,w] (4,)
            q: current joint positions in radians (7,)
            dq: current joint velocities in rad/s (7,)

        Returns:
            torques: (7,) joint torques in Nm
        """
        # Current EE pose
        ee_pos, ee_rot = self.model.fk(q)

        # 6D pose error
        error = pose_error(target_pos, target_quat_xyzw, ee_pos, ee_rot)

        # Desired EE twist (task-space proportional control)
        twist_desired = self.kp_task * error  # (6,)

        # Jacobian and damped pseudoinverse
        J = self.model.jacobian(q)  # (6, 7)
        JJT = J @ J.T + (self.damping ** 2) * np.eye(6)
        dq_desired = J.T @ np.linalg.solve(JJT, twist_desired)  # (7,)

        # Clamp desired joint velocities
        # dq_scale = np.max(np.abs(dq_desired)) / self.max_dq
        # if dq_scale > 1.0:
        #     dq_desired /= dq_scale

        # Desired joint position (one-step integration)
        q_desired = q + dq_desired * self.dt

        # Joint torques: PD tracking + gravity compensation
        dq_desired*=0.0
        # self.kp_joint*=0.0
        tau = (self.kp_joint * (q_desired - q)
               + self.kd_joint * (dq_desired - dq)
               + self.model.gravity(q))

        # Clamp torques
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        return tau
