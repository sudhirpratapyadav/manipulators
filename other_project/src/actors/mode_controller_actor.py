"""
ModeController Actor - handles position <-> torque mode transitions.

Responsibilities:
- Subscribe to mode change requests
- Execute stabilization sequences for mode transitions
- Coordinate with hardware for mode switching
- Publish mode changed notifications
"""

import queue
import time
from typing import Optional

from ..core.actor import Actor
from ..core.bus import MessageBus, Topics
from ..core.messages import (
    ModeChangeRequest, ModeChanged, ControlMode, RobotState
)
from ..core.config import Config


class ModeControllerActor(Actor):
    """
    Mode control actor - manages control mode transitions.

    Subscribes: /mode/change_request, /robot/state
    Publishes: /mode/changed, /control/mode (to hardware)
    """

    def __init__(self, bus: MessageBus, config: Config):
        super().__init__(
            name="ModeControllerActor",
            bus=bus,
            config=config
        )

        self._config = config

        # Queues
        self._request_queue: Optional[queue.Queue] = None
        self._state_queue: Optional[queue.Queue] = None

        # State
        self._current_mode = ControlMode.POSITION
        self._target_mode = ControlMode.POSITION
        self._latest_robot_state: Optional[RobotState] = None

        # Transition state
        self._in_transition = False
        self._stabilization_count = 0
        self._stabilization_needed = 50  # iterations at ~1kHz

    def setup(self) -> None:
        """Subscribe to request topics."""
        self._request_queue = self.bus.subscribe_queue(Topics.MODE_CHANGE_REQUEST, maxsize=5)
        self._state_queue = self.bus.subscribe_queue(Topics.ROBOT_STATE, maxsize=2)

    def loop(self) -> None:
        """Process mode change requests and handle transitions."""
        # Get latest robot state
        try:
            while True:
                self._latest_robot_state = self._state_queue.get_nowait()
        except queue.Empty:
            pass

        # Process requests
        try:
            while True:
                req: ModeChangeRequest = self._request_queue.get_nowait()
                if req.mode != self._target_mode:
                    print(f"[ModeController] Mode change request: {req.mode.value} ({req.reason})")
                    self._target_mode = req.mode
                    if req.mode == ControlMode.TORQUE:
                        self._stabilization_count = 0
                    self._in_transition = True
        except queue.Empty:
            pass

        # Handle transitions
        if self._in_transition:
            self._handle_transition()

        # Sleep to avoid busy-waiting (run at ~100Hz)
        time.sleep(0.01)

    def _handle_transition(self) -> None:
        """Execute mode transition with stabilization."""
        if self._target_mode == self._current_mode:
            self._in_transition = False
            return

        if self._target_mode == ControlMode.TORQUE:
            self._transition_to_torque()
        else:
            self._transition_to_position()

    def _transition_to_torque(self) -> None:
        """Transition to torque mode with stabilization."""
        if self._stabilization_count == 0:
            print("[ModeController] Starting transition to TORQUE mode")
            # Hardware will handle the low-level servoing mode change
            # We just coordinate the sequence

        # Note: Stabilization is now handled by hardware actor
        # This actor just manages the high-level coordination
        self._stabilization_count += 1

        # After stabilization period, confirm mode change
        if self._stabilization_count >= self._stabilization_needed:
            print("[ModeController] TORQUE mode active")
            self._current_mode = ControlMode.TORQUE
            self._in_transition = False

            # Publish mode changed notification
            self.bus.publish(Topics.MODE_CHANGED, ModeChanged(
                mode=ControlMode.TORQUE,
                previous_mode=ControlMode.POSITION
            ))

            # Also publish to hardware (for backwards compatibility)
            from ..core.messages import ControlModeChange
            self.bus.publish(Topics.CONTROL_MODE, ControlModeChange(
                mode=ControlMode.TORQUE,
                reason="Mode transition complete"
            ))

    def _transition_to_position(self) -> None:
        """Transition to position mode (immediate)."""
        print("[ModeController] Switching to POSITION mode")

        previous_mode = self._current_mode
        self._current_mode = ControlMode.POSITION
        self._in_transition = False
        self._stabilization_count = 0

        # Publish mode changed notification
        self.bus.publish(Topics.MODE_CHANGED, ModeChanged(
            mode=ControlMode.POSITION,
            previous_mode=previous_mode
        ))

        # Also publish to hardware (for backwards compatibility)
        from ..core.messages import ControlModeChange
        self.bus.publish(Topics.CONTROL_MODE, ControlModeChange(
            mode=ControlMode.POSITION,
            reason="Mode transition complete"
        ))

    def teardown(self) -> None:
        """Cleanup."""
        pass

    @property
    def current_mode(self) -> ControlMode:
        """Get current control mode."""
        return self._current_mode
