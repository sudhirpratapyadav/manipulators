"""
Control Actor - computes torque commands.

Responsibilities:
- Subscribe to RobotState and DesiredJoints
- Compute gravity compensation
- Compute PD control torques
- Publish TorqueCommand
"""

import queue
import numpy as np
from typing import Optional, Tuple

from ..core.actor import TimedActor
from ..core.bus import MessageBus, Topics
from ..core.messages import RobotState, TorqueCommand, DesiredJoints, ControlMode
from ..core.config import Config
from ..robot.model import RobotModel


class ControlActor(TimedActor):
    """
    Control computation actor (runs at hardware rate).

    Subscribes: /robot/state, /control/desired
    Publishes: /control/torque
    """

    def __init__(self, bus: MessageBus, config: Config):
        super().__init__(
            name="ControlActor",
            bus=bus,
            config=config,
            rate_hz=config.control.rates.hardware_hz
        )

        self._config = config
        self._model: Optional[RobotModel] = None

        # Gains from config
        self._kp = np.array(config.control.gains.kp)
        self._kd = np.array(config.control.gains.kd)
        self._max_pd_torque = config.control.limits.max_pd_torque_nm

        # State
        self._state_queue: Optional[queue.Queue] = None
        self._desired_queue: Optional[queue.Queue] = None
        self._enable_queue: Optional[queue.Queue] = None

        self._latest_state: Optional[RobotState] = None
        self._latest_desired: Optional[DesiredJoints] = None
        self._enabled = False

    def setup(self) -> None:
        """Initialize model and subscribe to topics."""
        self._model = RobotModel()
        self._state_queue = self.bus.subscribe_queue(Topics.ROBOT_STATE, maxsize=2)
        self._desired_queue = self.bus.subscribe_queue(Topics.CONTROL_DESIRED, maxsize=2)
        self._enable_queue = self.bus.subscribe_queue(Topics.CONTROL_ENABLE, maxsize=5)

    def enable(self) -> None:
        """Enable torque output (legacy method - prefer message-based)."""
        # Clear previous desired position so we start from current position
        self._latest_desired = None
        self._enabled = True
        print("[ControlActor] Enabled")

    def disable(self) -> None:
        """Disable torque output (legacy method - prefer message-based)."""
        self._enabled = False
        print("[ControlActor] Disabled")

    def loop(self) -> None:
        """Compute and publish torque command."""
        # Process enable/disable commands
        try:
            while True:
                from ..core.messages import EnableCommand
                msg: EnableCommand = self._enable_queue.get_nowait()
                if msg.enabled:
                    # Clear previous desired position
                    self._latest_desired = None
                    self._enabled = True
                    print(f"[ControlActor] Enabled: {msg.reason}")
                else:
                    self._enabled = False
                    print(f"[ControlActor] Disabled: {msg.reason}")
        except queue.Empty:
            pass

        # Get latest robot state
        try:
            while True:
                self._latest_state = self._state_queue.get_nowait()
        except queue.Empty:
            pass

        # Get latest desired joints
        try:
            while True:
                self._latest_desired = self._desired_queue.get_nowait()
        except queue.Empty:
            pass

        # Skip if not enabled or no state
        if not self._enabled or self._latest_state is None:
            return

        # Use current position as desired if no command received
        if self._latest_desired is None:
            q_desired = np.array(self._latest_state.joint_positions)
            using_current = True
        else:
            q_desired = np.array(self._latest_desired.positions)
            using_current = False

        q_current = np.array(self._latest_state.joint_positions)
        v_current = np.array(self._latest_state.joint_velocities)

        # Compute torques
        tau = self._compute_torque(q_current, v_current, q_desired)

        # Publish
        msg = TorqueCommand(torques=tuple(tau))
        self.bus.publish(Topics.CONTROL_TORQUE, msg)

    def _compute_torque(
        self,
        q: np.ndarray,
        v: np.ndarray,
        q_desired: np.ndarray
    ) -> np.ndarray:
        """
        Compute total torque: gravity compensation + PD control.

        Args:
            q: Current joint positions (rad)
            v: Current joint velocities (rad/s)
            q_desired: Desired joint positions (rad)

        Returns:
            Total joint torques (Nm)
        """
        # Gravity compensation
        tau_g = self._model.gravity(q)

        # PD control with angle wrapping
        pos_error = q_desired - q
        pos_error = (pos_error + np.pi) % (2 * np.pi) - np.pi

        tau_pd = self._kp * pos_error - self._kd * v
        tau_pd = np.clip(tau_pd, -self._max_pd_torque, self._max_pd_torque)

        return tau_g + tau_pd

    def teardown(self) -> None:
        """Cleanup."""
        self._enabled = False

    def set_desired(self, q_desired: np.ndarray, source: str = "direct") -> None:
        """
        Directly set desired joints (convenience method).

        Publishes to the bus so it goes through normal flow.
        """
        msg = DesiredJoints(positions=tuple(q_desired), source=source)
        self.bus.publish(Topics.CONTROL_DESIRED, msg)
