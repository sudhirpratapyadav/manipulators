"""
IK Actor - computes differential IK at 400Hz.

Responsibilities:
- Subscribe to RobotState and PoseDelta (from inputs)
- Accumulate pose deltas into target pose
- Compute differential IK to get desired joint positions
- Publish DesiredJoints
"""

import queue
import numpy as np
from typing import Optional, Tuple

from ..core.actor import TimedActor
from ..core.bus import MessageBus, Topics
from ..core.messages import RobotState, DesiredJoints, PoseDelta, TargetPose
from ..core.config import Config
from ..robot.model import RobotModel


class IKActor(TimedActor):
    """
    Differential IK computation actor (400Hz).

    Subscribes: /robot/state, /input/delta
    Publishes: /control/desired, /control/target_pose
    """

    def __init__(self, bus: MessageBus, config: Config):
        super().__init__(
            name="IKActor",
            bus=bus,
            config=config,
            rate_hz=config.control.rates.ik_hz
        )

        self._config = config
        self._model: Optional[RobotModel] = None

        # IK parameters from config
        ik_cfg = config.control.ik
        self._lambda_min = ik_cfg.lambda_min
        self._lambda_max = ik_cfg.lambda_max
        self._manip_threshold = ik_cfg.manip_threshold
        self._max_step = ik_cfg.max_step_rad

        # Position bounds
        self._pos_bound = config.control.limits.position_bound_m

        # State
        self._state_queue: Optional[queue.Queue] = None
        self._delta_queue: Optional[queue.Queue] = None
        self._enable_queue: Optional[queue.Queue] = None
        self._reset_queue: Optional[queue.Queue] = None

        self._latest_state: Optional[RobotState] = None
        self._target_pose: Optional[np.ndarray] = None
        self._home_pose: Optional[np.ndarray] = None
        self._home_q: Optional[np.ndarray] = None

        self._enabled = False

    def setup(self) -> None:
        """Initialize model and subscribe to topics."""
        self._model = RobotModel()
        self._state_queue = self.bus.subscribe_queue(Topics.ROBOT_STATE, maxsize=2)
        self._delta_queue = self.bus.subscribe_queue(Topics.INPUT_DELTA, maxsize=20)
        self._enable_queue = self.bus.subscribe_queue(Topics.IK_ENABLE, maxsize=5)
        self._reset_queue = self.bus.subscribe_queue(Topics.IK_RESET, maxsize=5)

        # Initialize home from config
        self._home_q = np.array(self._config.home_joints_rad)
        self._home_pose = self._model.forward_kinematics(self._home_q)
        self._target_pose = self._home_pose.copy()

    def enable(self, initial_q: Optional[np.ndarray] = None) -> None:
        """
        Enable IK computation (legacy method - prefer message-based).

        Args:
            initial_q: Initial joint configuration (uses current state if None)
        """
        if initial_q is not None:
            self._target_pose = self._model.forward_kinematics(initial_q)
        elif self._latest_state is not None:
            q = np.array(self._latest_state.joint_positions)
            self._target_pose = self._model.forward_kinematics(q)
        else:
            self._target_pose = self._home_pose.copy()

        self._enabled = True
        print(f"[IKActor] Enabled, target pose: {self._target_pose[:3]}")

    def disable(self) -> None:
        """Disable IK computation (legacy method - prefer message-based)."""
        self._enabled = False
        print("[IKActor] Disabled")

    def reset_to_home(self) -> None:
        """Reset target pose to home (legacy method - prefer message-based)."""
        self._target_pose = self._home_pose.copy()
        print("[IKActor] Reset to home pose")

    def loop(self) -> None:
        """Compute and publish desired joints."""
        # Process enable/disable commands
        try:
            while True:
                from ..core.messages import EnableCommand
                msg: EnableCommand = self._enable_queue.get_nowait()
                if msg.enabled:
                    # Extract initial_q if provided
                    initial_q = None
                    if "initial_q" in msg.initial_data:
                        initial_q = np.array(msg.initial_data["initial_q"])

                    # Set target pose
                    if initial_q is not None:
                        self._target_pose = self._model.forward_kinematics(initial_q)
                    elif self._latest_state is not None:
                        q = np.array(self._latest_state.joint_positions)
                        self._target_pose = self._model.forward_kinematics(q)
                    else:
                        self._target_pose = self._home_pose.copy()

                    self._enabled = True
                    print(f"[IKActor] Enabled: {msg.reason}, target pose: {self._target_pose[:3]}")
                else:
                    self._enabled = False
                    print(f"[IKActor] Disabled: {msg.reason}")
        except queue.Empty:
            pass

        # Process reset commands
        try:
            while True:
                from ..core.messages import ResetCommand
                msg: ResetCommand = self._reset_queue.get_nowait()
                self._target_pose = self._home_pose.copy()
                print("[IKActor] Reset to home pose")
        except queue.Empty:
            pass

        # Get latest robot state
        try:
            while True:
                self._latest_state = self._state_queue.get_nowait()
        except queue.Empty:
            pass

        if not self._enabled or self._latest_state is None:
            return

        # Accumulate pose deltas
        total_delta = np.zeros(6)
        try:
            while True:
                msg: PoseDelta = self._delta_queue.get_nowait()
                total_delta[:3] += np.array(msg.delta_position)
                total_delta[3:] += np.array(msg.delta_orientation)
        except queue.Empty:
            pass

        # Update target pose
        if np.linalg.norm(total_delta) > 1e-6:
            self._target_pose += total_delta

            # Clamp position to bounds around home
            pos_min = self._home_pose[:3] - self._pos_bound
            pos_max = self._home_pose[:3] + self._pos_bound
            self._target_pose[:3] = np.clip(self._target_pose[:3], pos_min, pos_max)

        # Compute IK
        q_current = np.array(self._latest_state.joint_positions)
        q_new, pos_err, rot_err = self._model.diff_ik_step(
            q_current,
            self._target_pose,
            max_step=self._max_step,
            lambda_min=self._lambda_min,
            lambda_max=self._lambda_max,
            manip_threshold=self._manip_threshold,
        )

        # Publish desired joints
        msg = DesiredJoints(positions=tuple(q_new), source="ik")
        self.bus.publish(Topics.CONTROL_DESIRED, msg)

        # Publish target pose for visualization
        target_msg = TargetPose(
            position=tuple(self._target_pose[:3]),
            orientation=tuple(self._target_pose[3:]),
        )
        self.bus.publish(Topics.TARGET_POSE, target_msg)

    def teardown(self) -> None:
        """Cleanup."""
        self._enabled = False

    @property
    def target_pose(self) -> Optional[np.ndarray]:
        return self._target_pose.copy() if self._target_pose is not None else None

    @property
    def home_pose(self) -> Optional[np.ndarray]:
        return self._home_pose.copy() if self._home_pose is not None else None

    @property
    def home_q(self) -> Optional[np.ndarray]:
        return self._home_q.copy() if self._home_q is not None else None
