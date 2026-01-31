"""
State Actor - aggregates state for UI display.

Responsibilities:
- Subscribe to all relevant state topics
- Aggregate into UIState message
- Publish at GUI rate (50Hz)
"""

import queue
import numpy as np
from typing import Optional
import pinocchio as pin

from ..core.actor import TimedActor
from ..core.bus import MessageBus, Topics
from ..core.messages import (
    RobotState, TargetPose, UIState, ControllerState, ControlMode, StateChange
)
from ..core.config import Config
from ..robot.model import RobotModel


class StateActor(TimedActor):
    """
    State aggregation actor for UI (50Hz).

    Subscribes: /robot/state, /control/target_pose, /system/state
    Publishes: /ui/state
    """

    def __init__(self, bus: MessageBus, config: Config):
        super().__init__(
            name="StateActor",
            bus=bus,
            config=config,
            rate_hz=config.control.rates.gui_hz
        )

        self._config = config
        self._model: Optional[RobotModel] = None

        # Queues
        self._state_queue: Optional[queue.Queue] = None
        self._target_queue: Optional[queue.Queue] = None
        self._sys_state_queue: Optional[queue.Queue] = None

        # Cached state
        self._latest_robot_state: Optional[RobotState] = None
        self._latest_target: Optional[TargetPose] = None
        self._controller_state = ControllerState.DISCONNECTED
        self._status_message = "Not connected"
        self._control_mode = ControlMode.POSITION

    def setup(self) -> None:
        """Subscribe to state topics."""
        self._model = RobotModel()
        self._state_queue = self.bus.subscribe_queue(Topics.ROBOT_STATE, maxsize=2)
        self._target_queue = self.bus.subscribe_queue(Topics.TARGET_POSE, maxsize=2)
        self._sys_state_queue = self.bus.subscribe_queue(Topics.SYSTEM_STATE, maxsize=5)

    def loop(self) -> None:
        """Aggregate state and publish UI update."""
        # Consume all pending messages
        try:
            while True:
                self._latest_robot_state = self._state_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self._latest_target = self._target_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                msg: StateChange = self._sys_state_queue.get_nowait()
                self._controller_state = msg.state
                self._status_message = msg.message
        except queue.Empty:
            pass

        # Build UI state
        if self._latest_robot_state is not None:
            q_rad = np.array(self._latest_robot_state.joint_positions)
            q_deg = np.degrees(q_rad)
            ee_pose = self._model.forward_kinematics(q_rad)
            ee_pos = tuple(ee_pose[:3])
            # Convert axis-angle to quaternion
            ee_rot_aa = ee_pose[3:6]
            ee_rot_mat = pin.exp3(ee_rot_aa)
            ee_quat = pin.Quaternion(ee_rot_mat)
            ee_orientation = (ee_quat.x, ee_quat.y, ee_quat.z, ee_quat.w)
            healthy = self._latest_robot_state.connection_healthy
            gripper_pos = self._latest_robot_state.gripper_position
        else:
            q_deg = np.zeros(7)
            ee_pos = (0.0, 0.0, 0.0)
            ee_orientation = (0.0, 0.0, 0.0, 1.0)
            healthy = False
            gripper_pos = 0.0

        if self._latest_target is not None:
            target_pos = self._latest_target.position
        else:
            target_pos = (0.0, 0.0, 0.0)

        ui_state = UIState(
            controller_state=self._controller_state,
            status_message=self._status_message,
            connection_healthy=healthy,
            joint_positions_deg=tuple(q_deg),
            ee_position=ee_pos,
            ee_orientation=ee_orientation,
            target_position=target_pos,
            control_mode=self._control_mode,
            gripper_position=gripper_pos,
        )

        self.bus.publish(Topics.UI_STATE, ui_state)

    def set_state(self, state: ControllerState, message: str = "") -> None:
        """
        Set controller state and publish change.

        This is called by the orchestrator to update state.
        """
        old_state = self._controller_state
        self._controller_state = state
        self._status_message = message

        change = StateChange(
            state=state,
            previous_state=old_state,
            message=message,
        )
        self.bus.publish(Topics.SYSTEM_STATE, change)

    def set_control_mode(self, mode: ControlMode) -> None:
        """Update tracked control mode."""
        self._control_mode = mode

    def get_current_robot_state(self) -> Optional[RobotState]:
        """Get the latest robot state (for initialization purposes)."""
        return self._latest_robot_state

    def teardown(self) -> None:
        """Cleanup."""
        pass
