"""Pose and rotation utilities."""

import numpy as np
import pinocchio as pin


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] (ROS convention) to 3x3 rotation matrix."""
    x, y, z, w = quat_xyzw
    return pin.Quaternion(w, x, y, z).toRotationMatrix()


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to quaternion [x,y,z,w] (ROS convention)."""
    q = pin.Quaternion(R)
    return np.array([q.x, q.y, q.z, q.w])


def orientation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Compute orientation error as a 3D vector (axis-angle via log map)."""
    R_err = R_target @ R_current.T
    return pin.log3(R_err)


def pose_error(
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    current_pos: np.ndarray,
    current_rot: np.ndarray,
) -> np.ndarray:
    """
    6D pose error: [position_error (3), orientation_error (3)].

    Args:
        target_pos: desired position (3,)
        target_quat_xyzw: desired orientation as quaternion [x,y,z,w]
        current_pos: current position (3,)
        current_rot: current orientation as 3x3 rotation matrix
    """
    e_pos = target_pos - current_pos
    R_target = quat_to_matrix(target_quat_xyzw)
    e_rot = orientation_error(R_target, current_rot)
    return np.concatenate([e_pos, e_rot])


def kinova_degrees_to_radians(positions_deg: np.ndarray) -> np.ndarray:
    """Convert Kinova's 0-360 degree range to radians (-pi to pi)."""
    signed = positions_deg.copy()
    signed[signed > 180.0] -= 360.0
    return np.deg2rad(signed)
