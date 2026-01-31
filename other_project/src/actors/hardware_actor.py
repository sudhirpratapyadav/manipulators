"""
Hardware Actor - handles all robot I/O at 1kHz.

Responsibilities:
- Read robot state from UDP
- Write torque commands to UDP
- Publish RobotState messages
- Subscribe to TorqueCommand messages
- Handle mode transitions (position <-> torque)
"""

import time
import queue
import numpy as np
from typing import Optional

from ..core.actor import TimedActor
from ..core.bus import MessageBus, Topics
from ..core.messages import (
    RobotState, TorqueCommand, ControlModeChange, ControlMode,
    ControllerState, StateChange
)
from ..core.config import Config
from ..robot.hardware_base import HardwareInterface, RobotFeedback

# Lazy imports for hardware backends to avoid loading kortex_api when not needed
# KinovaHardware and MuJoCoSimulator imported on-demand in connect()


class HardwareActor(TimedActor):
    """
    1kHz hardware I/O actor.

    Publishes: /robot/state (RobotState)
    Subscribes: /control/torque (TorqueCommand), /control/mode (ControlModeChange)
    """

    def __init__(self, bus: MessageBus, config: Config, scene_config=None):
        super().__init__(
            name="HardwareActor",
            bus=bus,
            config=config,
            rate_hz=config.control.rates.hardware_hz
        )

        self._config = config
        self._scene_config = scene_config
        self._hardware: Optional[HardwareInterface] = None

        # Queues for incoming commands
        self._torque_queue: Optional[queue.Queue] = None
        self._mode_queue: Optional[queue.Queue] = None
        self._desired_queue: Optional[queue.Queue] = None  # For joint control position commands
        self._gripper_queue: Optional[queue.Queue] = None  # For gripper commands

        # State
        self._current_mode = ControlMode.POSITION
        self._target_mode = ControlMode.POSITION
        self._last_feedback: Optional[RobotFeedback] = None
        self._consecutive_errors = 0
        self._max_errors = 10
        self._last_slider_command: Optional[np.ndarray] = None  # Last position command from sliders
        self._last_gripper_command: float = 0.0  # Last gripper position command

        # Mode transition state
        self._stabilization_count = 0
        self._stabilization_needed = 50  # iterations to stabilize before torque mode

    def setup(self) -> None:
        """Subscribe to command topics."""
        self._torque_queue = self.bus.subscribe_queue(Topics.CONTROL_TORQUE, maxsize=2)
        self._mode_queue = self.bus.subscribe_queue(Topics.CONTROL_MODE, maxsize=5)
        self._desired_queue = self.bus.subscribe_queue(Topics.CONTROL_DESIRED, maxsize=2)
        self._gripper_queue = self.bus.subscribe_queue(Topics.GRIPPER_COMMAND, maxsize=2)

    def set_scene_config(self, scene_config):
        """Set scene configuration (before connect)."""
        self._scene_config = scene_config

    def connect(self) -> bool:
        """
        Connect to robot hardware or simulation.

        Returns:
            True if successful
        """
        # Choose backend based on config - lazy import to avoid loading unused dependencies
        if self._config.simulation.enabled:
            print(f"[HardwareActor] Using MuJoCo simulation backend")
            from ..robot.mujoco_sim import MuJoCoSimulator
            self._hardware = MuJoCoSimulator(
                model_path=self._config.simulation.model_path,
                render=self._config.simulation.render,
                home_joints_rad=np.array(self._config.home_joints_rad),
                bus=self.bus,
                scene_config=self._scene_config,
            )
        else:
            print(f"[HardwareActor] Using real Kinova hardware backend")
            from ..robot.kinova import KinovaHardware
            self._hardware = KinovaHardware(
                ip=self._config.kinova.ip,
                username=self._config.kinova.username,
                password=self._config.kinova.password,
            )

        if not self._hardware.connect():
            self._hardware = None
            return False

        # Clear faults and set position mode
        self._hardware.clear_faults()
        time.sleep(0.3)
        self._hardware.set_servoing_mode(low_level=False)
        time.sleep(0.3)

        # Wait for arm ready
        if not self._hardware.wait_for_arm_ready(timeout=10.0):
            print("[HardwareActor] Arm not ready")
            self._hardware.disconnect()
            self._hardware = None
            return False

        # Read initial state
        self._last_feedback = self._hardware.read_feedback()
        if self._last_feedback is None:
            self._hardware.disconnect()
            self._hardware = None
            return False

        self._current_mode = ControlMode.POSITION
        self._target_mode = ControlMode.POSITION
        self._consecutive_errors = 0

        return True

    def disconnect(self) -> None:
        """Disconnect from robot hardware."""
        if self._hardware:
            self._hardware.disconnect()
            self._hardware = None

    def loop(self) -> None:
        """Main 1kHz loop iteration."""
        if self._hardware is None:
            time.sleep(0.1)
            return

        # Check for mode change requests
        self._process_mode_requests()

        # Handle mode transitions
        self._handle_mode_transition()

        # Read and publish state, send commands
        if self._current_mode == ControlMode.TORQUE:
            self._torque_loop()
        else:
            self._position_loop()

    def _process_mode_requests(self) -> None:
        """Process any pending mode change requests."""
        try:
            while True:
                msg: ControlModeChange = self._mode_queue.get_nowait()
                if msg.mode != self._target_mode:
                    print(f"[HardwareActor] Mode request: {msg.mode.value} ({msg.reason})")
                    self._target_mode = msg.mode
                    if msg.mode == ControlMode.TORQUE:
                        self._stabilization_count = 0
        except queue.Empty:
            pass

    def _handle_mode_transition(self) -> None:
        """Handle transitions between position and torque mode."""
        if self._target_mode == self._current_mode:
            return

        if self._target_mode == ControlMode.TORQUE:
            # Transition to torque mode
            if self._stabilization_count == 0:
                # Start transition
                print("[HardwareActor] Starting transition to TORQUE mode")
                self._hardware.set_servoing_mode(low_level=True)
                time.sleep(0.05)

            # Stabilize in position mode first
            if self._last_feedback and self._stabilization_count < self._stabilization_needed:
                self._hardware.send_positions(self._last_feedback.positions_deg)
                self._stabilization_count += 1
                return

            # Enable torque mode on actuators
            print("[HardwareActor] Enabling TORQUE mode on actuators")
            self._hardware.set_torque_mode(enabled=True)
            self._current_mode = ControlMode.TORQUE
            print("[HardwareActor] TORQUE mode active")

        else:
            # Transition to position mode
            print("[HardwareActor] Switching to POSITION mode")
            self._hardware.set_torque_mode(enabled=False)
            self._hardware.set_servoing_mode(low_level=False)
            self._current_mode = ControlMode.POSITION
            self._stabilization_count = 0
            print("[HardwareActor] POSITION mode active")

    def _position_loop(self) -> None:
        """Position mode: send position commands if available, otherwise just read."""
        # Check for new position command from sliders (joint control mode)
        try:
            while True:
                from ..core.messages import DesiredJoints
                msg: DesiredJoints = self._desired_queue.get_nowait()
                if msg.source == "sliders":  # Only handle slider commands in position mode
                    self._last_slider_command = np.degrees(np.array(msg.positions))
        except queue.Empty:
            pass

        # Check for new gripper command
        try:
            while True:
                from ..core.messages import GripperCommand
                msg: GripperCommand = self._gripper_queue.get_nowait()
                self._last_gripper_command = msg.position
        except queue.Empty:
            pass

        # Send position command if we have one, otherwise just read state
        if self._last_slider_command is not None:
            feedback = self._hardware.send_positions(self._last_slider_command, gripper_pos=self._last_gripper_command)
        else:
            # No command - just read current state
            # (Simulation runs independently, real hardware holds last target)
            feedback = self._hardware.read_feedback()

        if feedback:
            self._last_feedback = feedback
            self._publish_state(feedback, healthy=True)
            self._consecutive_errors = 0
        else:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_errors:
                self._publish_state(self._last_feedback, healthy=False)

    def _torque_loop(self) -> None:
        """Torque mode: get command, send torques, read feedback."""
        # Get latest torque command (non-blocking)
        torques = np.zeros(7)
        try:
            while True:
                msg: TorqueCommand = self._torque_queue.get_nowait()
                torques = np.array(msg.torques)
        except queue.Empty:
            pass

        # Send torques and read feedback
        if self._last_feedback is not None:
            feedback = self._hardware.send_torques(
                torques=torques,
                positions_deg=self._last_feedback.positions_deg
            )

            if feedback:
                self._last_feedback = feedback
                self._publish_state(feedback, healthy=True)
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1
                if self._consecutive_errors >= self._max_errors:
                    self._publish_state(self._last_feedback, healthy=False)
                    # Emergency: switch to position mode
                    self._target_mode = ControlMode.POSITION

    def _publish_state(self, feedback: Optional[RobotFeedback], healthy: bool) -> None:
        """Publish robot state to bus."""
        if feedback is None:
            return

        msg = RobotState(
            joint_positions=tuple(np.radians(feedback.positions_deg)),
            joint_velocities=tuple(np.radians(feedback.velocities_deg)),
            connection_healthy=healthy,
            gripper_position=feedback.gripper_position,
        )
        self.bus.publish(Topics.ROBOT_STATE, msg)

    def teardown(self) -> None:
        """Cleanup on shutdown."""
        # Ensure position mode before exit
        if self._hardware and self._current_mode == ControlMode.TORQUE:
            try:
                self._hardware.set_torque_mode(enabled=False)
                self._hardware.set_servoing_mode(low_level=False)
            except Exception:
                pass

    def execute_high_level_action(self, target_deg: np.ndarray, duration: float) -> bool:
        """
        Execute a high-level joint action (blocking).

        Used for "go to home" type movements.
        """
        if self._hardware is None:
            return False

        # Must be in position mode
        if self._current_mode == ControlMode.TORQUE:
            self._target_mode = ControlMode.POSITION
            time.sleep(0.3)

        self._hardware.clear_faults()
        time.sleep(0.2)
        self._hardware.set_servoing_mode(low_level=False)
        time.sleep(0.2)

        if not self._hardware.wait_for_arm_ready(timeout=5.0):
            return False

        return self._hardware.execute_joint_action(target_deg, duration)

    @property
    def is_connected(self) -> bool:
        return self._hardware is not None

    @property
    def current_mode(self) -> ControlMode:
        return self._current_mode

    def start_viewer(self):
        """Start visualization viewer (for simulation only)."""
        if self._hardware:
            self._hardware.start_viewer()

    def sync_viewer(self):
        """Sync viewer with current state (for simulation only)."""
        if self._hardware:
            self._hardware.sync_viewer()
