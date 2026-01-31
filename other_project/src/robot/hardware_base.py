"""
Abstract hardware interface for robot control.

Defines the common interface that both real hardware and simulation must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RobotFeedback:
    """Raw feedback from robot or simulation."""
    positions_deg: np.ndarray  # 7 values in degrees
    velocities_deg: np.ndarray  # 7 values in deg/s
    torques_measured: np.ndarray  # 7 values in Nm
    timestamp: float
    gripper_position: float = 0.0  # Gripper position (0-255 for MuJoCo, 0-1 for real hardware)


class HardwareInterface(ABC):
    """
    Abstract interface for robot hardware or simulation backend.

    All methods must be implemented by concrete classes (KinovaHardware, MuJoCoSimulator).
    This ensures both backends can be used interchangeably.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to hardware or load simulation.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        pass

    @abstractmethod
    def clear_faults(self) -> None:
        """Clear any robot faults (no-op for simulation)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop all robot motion."""
        pass

    @abstractmethod
    def set_servoing_mode(self, low_level: bool) -> None:
        """
        Set robot servoing mode.

        Args:
            low_level: True for LOW_LEVEL_SERVOING, False for SINGLE_LEVEL_SERVOING
        """
        pass

    @abstractmethod
    def set_torque_mode(self, enabled: bool) -> None:
        """
        Enable or disable torque control mode.

        Args:
            enabled: True for TORQUE mode, False for POSITION mode
        """
        pass

    @abstractmethod
    def is_arm_ready(self) -> bool:
        """Check if arm is in ready state."""
        pass

    @abstractmethod
    def wait_for_arm_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait for arm to be in ready state.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if arm is ready, False if timeout
        """
        pass

    @abstractmethod
    def read_feedback(self) -> Optional[RobotFeedback]:
        """
        Read current robot state.

        Returns:
            RobotFeedback or None if error
        """
        pass

    @abstractmethod
    def send_torques(
        self,
        torques: np.ndarray,
        positions_deg: np.ndarray
    ) -> Optional[RobotFeedback]:
        """
        Send torque command and read feedback.

        Args:
            torques: Joint torques (7,) in Nm
            positions_deg: Current positions for command echo (7,) in degrees

        Returns:
            RobotFeedback or None if error
        """
        pass

    @abstractmethod
    def send_positions(self, positions_deg: np.ndarray) -> Optional[RobotFeedback]:
        """
        Send position command (for stabilization before torque mode).

        Args:
            positions_deg: Joint positions (7,) in degrees

        Returns:
            RobotFeedback or None if error
        """
        pass

    @abstractmethod
    def execute_joint_action(
        self,
        target_positions_deg: np.ndarray,
        duration: float = 8.0,
        callback=None,
    ) -> bool:
        """
        Execute high-level joint position action.

        Args:
            target_positions_deg: Target joint angles (7,) in degrees
            duration: Motion duration in seconds
            callback: Optional callback for progress

        Returns:
            True if successful
        """
        pass

    @property
    @abstractmethod
    def in_torque_mode(self) -> bool:
        """Check if currently in torque mode."""
        pass

    def start_viewer(self):
        """
        Start visualization viewer (optional, for simulation only).

        Default implementation does nothing (for real hardware).
        Can be overridden by simulation backends.
        """
        pass

    def sync_viewer(self):
        """
        Sync viewer with current state (optional, for simulation only).

        Should be called periodically from main event loop.
        Default implementation does nothing (for real hardware).
        """
        pass
