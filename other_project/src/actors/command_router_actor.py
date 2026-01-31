"""
CommandRouter Actor - orchestrates high-level system commands.

Responsibilities:
- Subscribe to SystemCommand messages from GUI
- Break down high-level commands into sequences of low-level actions
- Coordinate multi-step operations (go home, start diff-IK, etc.)
- Publish state transition requests and mode change requests
"""

import queue
import time
import numpy as np
from typing import Optional

from ..core.actor import Actor
from ..core.bus import MessageBus, Topics
from ..core.messages import (
    SystemCommand, SystemCommandType, StateTransitionRequest,
    ControllerState, ModeChangeRequest, ControlMode,
    EnableCommand, DesiredJoints, RobotState
)
from ..core.config import Config


class CommandRouterActor(Actor):
    """
    Command routing and orchestration actor.

    Subscribes: /system/command, /robot/state, /system/state, /mode/changed
    Publishes: /state/transition_request, /mode/change_request, /control/enable,
               /ik/enable, /input/enable, /control/desired
    """

    def __init__(self, bus: MessageBus, config: Config):
        super().__init__(
            name="CommandRouterActor",
            bus=bus,
            config=config
        )

        self._config = config

        # Queues
        self._command_queue: Optional[queue.Queue] = None
        self._state_queue: Optional[queue.Queue] = None
        self._robot_state_queue: Optional[queue.Queue] = None
        self._mode_changed_queue: Optional[queue.Queue] = None

        # Current state tracking
        self._current_state = ControllerState.DISCONNECTED
        self._latest_robot_state: Optional[RobotState] = None

        # Multi-step operation tracking
        self._pending_operation: Optional[str] = None
        self._operation_start_time: float = 0.0

    def setup(self) -> None:
        """Subscribe to command topics."""
        self._command_queue = self.bus.subscribe_queue(Topics.SYSTEM_COMMAND, maxsize=10)
        self._state_queue = self.bus.subscribe_queue(Topics.SYSTEM_STATE, maxsize=5)
        self._robot_state_queue = self.bus.subscribe_queue(Topics.ROBOT_STATE, maxsize=2)
        self._mode_changed_queue = self.bus.subscribe_queue(Topics.MODE_CHANGED, maxsize=5)

    def loop(self) -> None:
        """Process commands and coordinate operations."""
        # Update current state
        try:
            while True:
                from ..core.messages import StateChange
                msg: StateChange = self._state_queue.get_nowait()
                self._current_state = msg.state
        except queue.Empty:
            pass

        # Update robot state
        try:
            while True:
                self._latest_robot_state = self._robot_state_queue.get_nowait()
        except queue.Empty:
            pass

        # Process mode changes (for multi-step operations)
        try:
            while True:
                from ..core.messages import ModeChanged
                msg: ModeChanged = self._mode_changed_queue.get_nowait()
                self._on_mode_changed(msg)
        except queue.Empty:
            pass

        # Process commands
        try:
            while True:
                cmd: SystemCommand = self._command_queue.get_nowait()
                self._handle_command(cmd)
        except queue.Empty:
            pass

        # Update pending operations
        self._update_pending_operations()

        # Sleep to avoid busy-waiting
        time.sleep(0.01)

    def _handle_command(self, cmd: SystemCommand) -> None:
        """Route system command to appropriate handler."""
        print(f"[CommandRouter] Received command: {cmd.command.value}")

        handlers = {
            SystemCommandType.CONNECT: self._handle_connect,
            SystemCommandType.DISCONNECT: self._handle_disconnect,
            SystemCommandType.GO_HOME: self._handle_go_home,
            SystemCommandType.START_JOINT_CONTROL: self._handle_start_joint_control,
            SystemCommandType.START_DIFFIK: self._handle_start_diffik,
            SystemCommandType.STOP: self._handle_stop,
            SystemCommandType.EMERGENCY_STOP: self._handle_emergency_stop,
        }

        handler = handlers.get(cmd.command)
        if handler:
            handler(cmd)
        else:
            print(f"[CommandRouter] Unknown command: {cmd.command}")

    def _handle_connect(self, cmd: SystemCommand) -> None:
        """Handle connect command."""
        if self._current_state != ControllerState.DISCONNECTED:
            print("[CommandRouter] Already connected or connecting")
            return

        # Request state transition
        self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
            target_state=ControllerState.CONNECTING,
            reason="Connection requested"
        ))

    def _handle_disconnect(self, cmd: SystemCommand) -> None:
        """Handle disconnect command."""
        # First stop any active control
        self._handle_stop(cmd)

        # Request disconnection
        self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
            target_state=ControllerState.DISCONNECTED,
            reason="Disconnection requested"
        ))

    def _handle_go_home(self, cmd: SystemCommand) -> None:
        """Handle go home command."""
        if self._current_state != ControllerState.IDLE:
            print(f"[CommandRouter] Cannot go home from state: {self._current_state.value}")
            return

        # Transition to going home state
        self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
            target_state=ControllerState.GOING_HOME,
            reason="Going to home position"
        ))

        # Publish home position as desired
        home_q = np.array(self._config.home_joints_rad)
        self.bus.publish(Topics.CONTROL_DESIRED, DesiredJoints(
            positions=tuple(home_q),
            source="go_home"
        ))

        # Track operation
        self._pending_operation = "go_home"
        self._operation_start_time = time.time()

    def _handle_start_joint_control(self, cmd: SystemCommand) -> None:
        """Handle start joint control command."""
        if self._current_state != ControllerState.IDLE:
            print(f"[CommandRouter] Cannot start joint control from state: {self._current_state.value}")
            return

        # Transition to joint control state
        self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
            target_state=ControllerState.JOINT_CONTROL,
            reason="Joint control mode started"
        ))

        # Joint control stays in position mode - sliders will send position commands

    def _handle_start_diffik(self, cmd: SystemCommand) -> None:
        """Handle start differential IK command."""
        if self._current_state != ControllerState.IDLE:
            print(f"[CommandRouter] Cannot start Diff-IK from state: {self._current_state.value}")
            return

        # Transition to initialization state
        self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
            target_state=ControllerState.DIFFIK_INIT,
            reason="Initializing Diff-IK"
        ))

        # Start multi-step initialization
        self._pending_operation = "diffik_init"
        self._operation_start_time = time.time()

        # Step 1: Go to home position first
        home_q = np.array(self._config.home_joints_rad)
        self.bus.publish(Topics.CONTROL_DESIRED, DesiredJoints(
            positions=tuple(home_q),
            source="diffik_init"
        ))

    def _handle_stop(self, cmd: SystemCommand) -> None:
        """Handle stop command."""
        print("[CommandRouter] Stop requested")

        # Disable all control
        self.bus.publish(Topics.INPUT_ENABLE, EnableCommand(
            enabled=False,
            reason="Stop requested"
        ))
        self.bus.publish(Topics.IK_ENABLE, EnableCommand(
            enabled=False,
            reason="Stop requested"
        ))
        self.bus.publish(Topics.CONTROL_ENABLE, EnableCommand(
            enabled=False,
            reason="Stop requested"
        ))

        # Switch to position mode
        self.bus.publish(Topics.MODE_CHANGE_REQUEST, ModeChangeRequest(
            mode=ControlMode.POSITION,
            reason="Stop requested"
        ))

        # Clear pending operations
        self._pending_operation = None

        # Return to idle if we're in an active state
        if self._current_state not in (ControllerState.DISCONNECTED, ControllerState.CONNECTING):
            self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
                target_state=ControllerState.IDLE,
                reason="Stopped"
            ))

    def _handle_emergency_stop(self, cmd: SystemCommand) -> None:
        """Handle emergency stop command."""
        print("[CommandRouter] EMERGENCY STOP")

        # Immediately disable everything
        self._handle_stop(cmd)

        # Transition to error state
        self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
            target_state=ControllerState.ERROR,
            reason="Emergency stop activated"
        ))

    def _update_pending_operations(self) -> None:
        """Update multi-step pending operations."""
        if self._pending_operation is None:
            return

        if self._pending_operation == "go_home":
            self._update_go_home()
        elif self._pending_operation == "diffik_init":
            self._update_diffik_init()

    def _update_go_home(self) -> None:
        """Monitor go home operation progress."""
        # Simple timeout-based completion (in real system, check position convergence)
        elapsed = time.time() - self._operation_start_time
        if elapsed > 8.0:  # Match duration from original code
            print("[CommandRouter] Go home complete")
            self._pending_operation = None

            # Return to idle
            self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
                target_state=ControllerState.IDLE,
                reason="Home position reached"
            ))

    def _update_diffik_init(self) -> None:
        """Monitor Diff-IK initialization progress."""
        elapsed = time.time() - self._operation_start_time

        # Wait for home position (8 seconds like go_home)
        if elapsed > 8.0 and self._current_state == ControllerState.DIFFIK_INIT:
            print("[CommandRouter] Diff-IK initialization complete")

            # Get current position for IK initialization
            if self._latest_robot_state is not None:
                home_q = np.array(self._latest_robot_state.joint_positions)
            else:
                home_q = np.array(self._config.home_joints_rad)

            # Enable IK with initial position
            self.bus.publish(Topics.IK_ENABLE, EnableCommand(
                enabled=True,
                initial_data={"initial_q": home_q.tolist()},
                reason="Diff-IK started"
            ))

            # Enable control
            self.bus.publish(Topics.CONTROL_ENABLE, EnableCommand(
                enabled=True,
                reason="Diff-IK started"
            ))

            # Enable keyboard input
            self.bus.publish(Topics.INPUT_ENABLE, EnableCommand(
                enabled=True,
                reason="Diff-IK started"
            ))

            # Request torque mode
            self.bus.publish(Topics.MODE_CHANGE_REQUEST, ModeChangeRequest(
                mode=ControlMode.TORQUE,
                reason="Diff-IK started"
            ))

            self._pending_operation = "diffik_wait_torque"

    def _on_mode_changed(self, msg) -> None:
        """Handle mode change notifications."""
        if self._pending_operation == "diffik_wait_torque" and msg.mode == ControlMode.TORQUE:
            print("[CommandRouter] Diff-IK fully active")

            # Transition to active state
            self.bus.publish(Topics.STATE_TRANSITION_REQUEST, StateTransitionRequest(
                target_state=ControllerState.DIFFIK_ACTIVE,
                reason="Diff-IK active"
            ))

            self._pending_operation = None

    def teardown(self) -> None:
        """Cleanup."""
        pass
