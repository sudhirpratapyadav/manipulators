"""
Safety Actor - monitors robot state for safety violations.

Responsibilities:
- Subscribe to RobotState
- Check velocity limits
- Check position bounds (optional)
- Publish SafetyEvent on violations
- Can trigger emergency stop
"""

import queue
import numpy as np
from typing import Optional

from ..core.actor import TimedActor
from ..core.bus import MessageBus, Topics
from ..core.messages import RobotState, SafetyEvent, ControlModeChange, ControlMode
from ..core.config import Config


class SafetyActor(TimedActor):
    """
    Safety monitoring actor (runs at hardware rate).

    Subscribes: /robot/state
    Publishes: /safety/event, /control/mode (on emergency)
    """

    def __init__(self, bus: MessageBus, config: Config):
        super().__init__(
            name="SafetyActor",
            bus=bus,
            config=config,
            rate_hz=config.control.rates.safety_hz
        )

        self._config = config

        # Safety limits
        self._max_velocity_rad_s = np.radians(config.control.limits.max_velocity_deg_s)

        # State
        self._state_queue: Optional[queue.Queue] = None
        self._enabled = True
        self._violation_count = 0
        self._max_violations_before_stop = 3

    def setup(self) -> None:
        """Subscribe to robot state."""
        self._state_queue = self.bus.subscribe_queue(Topics.ROBOT_STATE, maxsize=2)

    def loop(self) -> None:
        """Check safety constraints."""
        if not self._enabled:
            return

        # Get latest state
        state: Optional[RobotState] = None
        try:
            while True:
                state = self._state_queue.get_nowait()
        except queue.Empty:
            pass

        if state is None:
            return

        # Check velocity limits
        velocities = np.array(state.joint_velocities)
        max_vel = np.max(np.abs(velocities))

        if max_vel > self._max_velocity_rad_s:
            self._violation_count += 1
            joint_idx = np.argmax(np.abs(velocities))

            event = SafetyEvent(
                event_type="velocity_limit",
                details=f"Joint {joint_idx + 1} velocity {np.degrees(velocities[joint_idx]):.1f} deg/s exceeds limit",
                severity="error" if self._violation_count >= self._max_violations_before_stop else "warning",
            )
            self.bus.publish(Topics.SAFETY_EVENT, event)

            if self._violation_count >= self._max_violations_before_stop:
                print(f"[SafetyActor] EMERGENCY STOP: Velocity limit exceeded")
                self._trigger_emergency_stop()
        else:
            # Reset violation count on clean iteration
            if self._violation_count > 0:
                self._violation_count = max(0, self._violation_count - 1)

        # Check connection health
        if not state.connection_healthy:
            event = SafetyEvent(
                event_type="connection_lost",
                details="Robot connection unhealthy",
                severity="critical",
            )
            self.bus.publish(Topics.SAFETY_EVENT, event)
            self._trigger_emergency_stop()

    def _trigger_emergency_stop(self) -> None:
        """Trigger emergency stop by switching to position mode."""
        msg = ControlModeChange(
            mode=ControlMode.POSITION,
            reason="Safety violation - emergency stop"
        )
        self.bus.publish(Topics.CONTROL_MODE, msg)

    def enable(self) -> None:
        """Enable safety monitoring."""
        self._enabled = True
        self._violation_count = 0

    def disable(self) -> None:
        """Disable safety monitoring (use with caution!)."""
        self._enabled = False

    def teardown(self) -> None:
        """Cleanup."""
        pass
