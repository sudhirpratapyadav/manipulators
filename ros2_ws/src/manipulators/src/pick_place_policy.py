#!/usr/bin/env python3
"""
Reactive Pick-and-Place Policy Node

A finite state machine policy that performs pick-and-place operations
with smooth, velocity-limited motion. The policy is reactive - it
continuously updates targets based on object detection.

States:
    INIT → IDLE → APPROACH → DESCEND → GRASP → LIFT → MID_TRANSPORT →
    PRE_PLACE → PLACE_DESCEND → RELEASE → PLACE_ASCEND → RETURN_IDLE → IDLE

Subscriptions:
    /detected_object_point (geometry_msgs/PointStamped): Object position in robot frame
    /ee_pose (geometry_msgs/PoseStamped): Current end-effector pose

Publications:
    /target_pose (geometry_msgs/PoseStamped): Target pose for control node
    /gripper_command (std_msgs/Float64): Gripper command (0=open, 1=closed)

Services:
    /pick_place/start (std_srvs/Trigger): Start pick-place cycle
    /pick_place/abort (std_srvs/Trigger): Abort current operation
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import threading

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Float64
from std_srvs.srv import Trigger


class State(Enum):
    """Pick-place state machine states."""
    INIT = auto()
    IDLE = auto()
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    MID_TRANSPORT = auto()
    PRE_PLACE = auto()
    PLACE_DESCEND = auto()
    RELEASE = auto()
    PLACE_ASCEND = auto()
    RETURN_IDLE = auto()


@dataclass
class PolicyConfig:
    """Configuration parameters for pick-place policy."""
    # Positions
    idle_position: np.ndarray
    mid_transport_position: np.ndarray
    pre_place_position: np.ndarray
    place_position: np.ndarray

    # Heights
    pre_grasp_height: float
    grasp_height: float
    lift_height: float

    # Orientation
    grasp_orientation: np.ndarray  # quaternion xyzw

    # Motion limits
    max_linear_velocity: float
    max_angular_velocity: float
    position_threshold: float
    orientation_threshold: float

    # Timing
    grasp_settle_time: float
    release_settle_time: float
    detection_timeout: float
    policy_rate: float


class PickPlacePolicy(Node):
    """Reactive pick-and-place policy using FSM with velocity-limited motion."""

    def __init__(self):
        super().__init__('pick_place_policy')

        # Load configuration
        self._config = self._load_config()

        # State machine
        self._state = State.INIT
        self._state_start_time: Optional[Time] = None
        self._lock = threading.Lock()

        # Current targets (smoothly updated)
        self._current_target_pos = self._config.idle_position.copy()
        self._current_target_quat = self._config.grasp_orientation.copy()
        self._gripper_command = 0.0  # 0=open, 1=closed

        # Sensor data
        self._ee_pos: Optional[np.ndarray] = None
        self._ee_quat: Optional[np.ndarray] = None
        self._object_pos: Optional[np.ndarray] = None
        self._object_detection_time: Optional[Time] = None
        self._last_valid_object_pos: Optional[np.ndarray] = None

        # Grasp position (latched when entering GRASP state)
        self._grasp_pos: Optional[np.ndarray] = None

        # Subscribers
        self._object_sub = self.create_subscription(
            PointStamped,
            '/detected_object_point',
            self._object_callback,
            10
        )
        self._ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/ee_pose',
            self._ee_pose_callback,
            10
        )

        # Publishers
        self._target_pose_pub = self.create_publisher(PoseStamped, '/target_pose', 10)
        self._gripper_pub = self.create_publisher(Float64, '/gripper_command', 10)

        # Services
        self._start_srv = self.create_service(
            Trigger,
            '/pick_place/start',
            self._start_callback
        )
        self._abort_srv = self.create_service(
            Trigger,
            '/pick_place/abort',
            self._abort_callback
        )

        # Main policy loop timer
        period = 1.0 / self._config.policy_rate
        self._timer = self.create_timer(period, self._policy_loop)

        self.get_logger().info(
            f'Pick-place policy initialized. Waiting for EE pose to move to idle position...'
        )

    def _load_config(self) -> PolicyConfig:
        """Load configuration from parameters."""
        self.declare_parameter('pick_place.idle_position', [0.4, 0.0, 0.4])
        self.declare_parameter('pick_place.mid_transport_position', [0.4, 0.0, 0.35])
        self.declare_parameter('pick_place.pre_place_position', [0.4, 0.3, 0.25])
        self.declare_parameter('pick_place.place_position', [0.4, 0.3, 0.05])
        self.declare_parameter('pick_place.pre_grasp_height', 0.10)
        self.declare_parameter('pick_place.grasp_height', 0.025)
        self.declare_parameter('pick_place.lift_height', 0.15)
        self.declare_parameter('pick_place.grasp_orientation', [0.0, 0.7071, 0.0, 0.7071])
        self.declare_parameter('pick_place.max_linear_velocity', 0.25)
        self.declare_parameter('pick_place.max_angular_velocity', 1.0)
        self.declare_parameter('pick_place.position_threshold', 0.01)
        self.declare_parameter('pick_place.orientation_threshold', 0.05)
        self.declare_parameter('pick_place.grasp_settle_time', 0.8)
        self.declare_parameter('pick_place.release_settle_time', 0.5)
        self.declare_parameter('pick_place.detection_timeout', 1.0)
        self.declare_parameter('pick_place.policy_rate', 50.0)

        return PolicyConfig(
            idle_position=np.array(
                self.get_parameter('pick_place.idle_position').value, dtype=np.float64
            ),
            mid_transport_position=np.array(
                self.get_parameter('pick_place.mid_transport_position').value, dtype=np.float64
            ),
            pre_place_position=np.array(
                self.get_parameter('pick_place.pre_place_position').value, dtype=np.float64
            ),
            place_position=np.array(
                self.get_parameter('pick_place.place_position').value, dtype=np.float64
            ),
            pre_grasp_height=self.get_parameter('pick_place.pre_grasp_height').value,
            grasp_height=self.get_parameter('pick_place.grasp_height').value,
            lift_height=self.get_parameter('pick_place.lift_height').value,
            grasp_orientation=np.array(
                self.get_parameter('pick_place.grasp_orientation').value, dtype=np.float64
            ),
            max_linear_velocity=self.get_parameter('pick_place.max_linear_velocity').value,
            max_angular_velocity=self.get_parameter('pick_place.max_angular_velocity').value,
            position_threshold=self.get_parameter('pick_place.position_threshold').value,
            orientation_threshold=self.get_parameter('pick_place.orientation_threshold').value,
            grasp_settle_time=self.get_parameter('pick_place.grasp_settle_time').value,
            release_settle_time=self.get_parameter('pick_place.release_settle_time').value,
            detection_timeout=self.get_parameter('pick_place.detection_timeout').value,
            policy_rate=self.get_parameter('pick_place.policy_rate').value,
        )

    def _object_callback(self, msg: PointStamped):
        """Handle object detection updates."""
        with self._lock:
            self._object_pos = np.array([msg.point.x, msg.point.y, msg.point.z])
            self._object_detection_time = self.get_clock().now()
            self._last_valid_object_pos = self._object_pos.copy()

    def _ee_pose_callback(self, msg: PoseStamped):
        """Handle end-effector pose updates."""
        with self._lock:
            self._ee_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            self._ee_quat = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])

    def _start_callback(self, request, response):
        """Handle start service request."""
        with self._lock:
            # Check if we're in IDLE state
            if self._state != State.IDLE:
                response.success = False
                response.message = f'Cannot start: currently in {self._state.name} state'
                return response

            # Check for fresh object detection
            if not self._is_detection_fresh():
                response.success = False
                response.message = 'Cannot start: no fresh object detection'
                return response

            # Start the pick-place cycle
            self._transition_to(State.APPROACH)
            response.success = True
            response.message = 'Pick-place cycle started'
            self.get_logger().info('Pick-place cycle started')

        return response

    def _abort_callback(self, request, response):
        """Handle abort service request."""
        with self._lock:
            if self._state in (State.INIT, State.IDLE):
                response.success = False
                response.message = 'Nothing to abort'
                return response

            # Open gripper and return to idle
            self._gripper_command = 0.0
            self._transition_to(State.RETURN_IDLE)
            response.success = True
            response.message = 'Aborting, returning to idle'
            self.get_logger().warn('Pick-place aborted by user')

        return response

    def _is_detection_fresh(self) -> bool:
        """Check if we have a recent object detection."""
        if self._object_detection_time is None:
            return False
        age = (self.get_clock().now() - self._object_detection_time).nanoseconds / 1e9
        return age < self._config.detection_timeout

    def _get_object_position(self) -> Optional[np.ndarray]:
        """Get current object position (latest or last known)."""
        if self._object_pos is not None:
            return self._object_pos.copy()
        return self._last_valid_object_pos.copy() if self._last_valid_object_pos is not None else None

    def _transition_to(self, new_state: State):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._state_start_time = self.get_clock().now()

        # State entry actions
        if new_state == State.GRASP:
            self._gripper_command = 1.0  # Close gripper
            # Latch the grasp position for lift calculation
            if self._last_valid_object_pos is not None:
                self._grasp_pos = self._last_valid_object_pos.copy()
        elif new_state == State.RELEASE:
            self._gripper_command = 0.0  # Open gripper
        elif new_state == State.IDLE:
            # Clear object detection for next cycle
            self._object_pos = None
            self._object_detection_time = None
            self._grasp_pos = None

        self.get_logger().info(f'State: {old_state.name} → {new_state.name}')

    def _time_in_state(self) -> float:
        """Get time spent in current state (seconds)."""
        if self._state_start_time is None:
            return 0.0
        return (self.get_clock().now() - self._state_start_time).nanoseconds / 1e9

    def _position_reached(self, target: np.ndarray) -> bool:
        """Check if end-effector has reached target position."""
        if self._ee_pos is None:
            return False
        error = np.linalg.norm(target - self._ee_pos)
        return error < self._config.position_threshold

    def _compute_desired_target(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the desired target position/orientation for current state.

        Returns:
            (position, orientation_quat, gripper_command)
        """
        cfg = self._config
        obj_pos = self._get_object_position()

        if self._state == State.INIT:
            return cfg.idle_position, cfg.grasp_orientation, 0.0

        elif self._state == State.IDLE:
            return cfg.idle_position, cfg.grasp_orientation, 0.0

        elif self._state == State.APPROACH:
            # Above object at pre-grasp height
            if obj_pos is not None:
                target = obj_pos.copy()
                target[2] += cfg.pre_grasp_height
                return target, cfg.grasp_orientation, 0.0
            return self._current_target_pos, cfg.grasp_orientation, 0.0

        elif self._state == State.DESCEND:
            # At object grasp height
            if obj_pos is not None:
                target = obj_pos.copy()
                target[2] += cfg.grasp_height
                return target, cfg.grasp_orientation, 0.0
            return self._current_target_pos, cfg.grasp_orientation, 0.0

        elif self._state == State.GRASP:
            # Hold position while gripper closes
            return self._current_target_pos, cfg.grasp_orientation, 1.0

        elif self._state == State.LIFT:
            # Lift above grasp position
            if self._grasp_pos is not None:
                target = self._grasp_pos.copy()
                target[2] += cfg.lift_height
                return target, cfg.grasp_orientation, 1.0
            return self._current_target_pos, cfg.grasp_orientation, 1.0

        elif self._state == State.MID_TRANSPORT:
            return cfg.mid_transport_position, cfg.grasp_orientation, 1.0

        elif self._state == State.PRE_PLACE:
            return cfg.pre_place_position, cfg.grasp_orientation, 1.0

        elif self._state == State.PLACE_DESCEND:
            return cfg.place_position, cfg.grasp_orientation, 1.0

        elif self._state == State.RELEASE:
            # Hold position while gripper opens
            return self._current_target_pos, cfg.grasp_orientation, 0.0

        elif self._state == State.PLACE_ASCEND:
            return cfg.pre_place_position, cfg.grasp_orientation, 0.0

        elif self._state == State.RETURN_IDLE:
            return cfg.idle_position, cfg.grasp_orientation, 0.0

        # Fallback
        return self._current_target_pos, cfg.grasp_orientation, self._gripper_command

    def _update_target_with_velocity_limit(
        self,
        desired_pos: np.ndarray,
        desired_quat: np.ndarray,
        dt: float
    ):
        """
        Update current target toward desired, respecting velocity limits.

        This creates smooth motion - the target moves at most max_velocity * dt
        per step, preventing sudden jumps.
        """
        # Position update with velocity limit
        pos_diff = desired_pos - self._current_target_pos
        pos_dist = np.linalg.norm(pos_diff)
        max_pos_step = self._config.max_linear_velocity * dt

        if pos_dist > max_pos_step:
            # Move toward target at max velocity
            self._current_target_pos += (pos_diff / pos_dist) * max_pos_step
        else:
            # Close enough, snap to target
            self._current_target_pos = desired_pos.copy()

        # Orientation update with velocity limit (simple SLERP-like approach)
        # For simplicity, we just snap orientation (usually doesn't change much)
        # A full implementation would use quaternion SLERP with angular velocity limit
        quat_diff = np.linalg.norm(desired_quat - self._current_target_quat)
        max_quat_step = self._config.max_angular_velocity * dt * 0.5  # approximate

        if quat_diff > max_quat_step:
            # Interpolate toward target
            alpha = max_quat_step / quat_diff
            self._current_target_quat = (
                (1 - alpha) * self._current_target_quat + alpha * desired_quat
            )
            # Renormalize quaternion
            self._current_target_quat /= np.linalg.norm(self._current_target_quat)
        else:
            self._current_target_quat = desired_quat.copy()

    def _check_transitions(self):
        """Check and execute state transitions."""
        cfg = self._config

        if self._state == State.INIT:
            if self._position_reached(cfg.idle_position):
                self._transition_to(State.IDLE)

        elif self._state == State.IDLE:
            # Transitions handled by service callback
            pass

        elif self._state == State.APPROACH:
            obj_pos = self._get_object_position()
            if obj_pos is not None:
                pre_grasp = obj_pos.copy()
                pre_grasp[2] += cfg.pre_grasp_height
                if self._position_reached(pre_grasp):
                    self._transition_to(State.DESCEND)

        elif self._state == State.DESCEND:
            obj_pos = self._get_object_position()
            if obj_pos is not None:
                grasp_pos = obj_pos.copy()
                grasp_pos[2] += cfg.grasp_height
                if self._position_reached(grasp_pos):
                    self._transition_to(State.GRASP)

        elif self._state == State.GRASP:
            if self._time_in_state() >= cfg.grasp_settle_time:
                self._transition_to(State.LIFT)

        elif self._state == State.LIFT:
            if self._grasp_pos is not None:
                lift_pos = self._grasp_pos.copy()
                lift_pos[2] += cfg.lift_height
                if self._position_reached(lift_pos):
                    self._transition_to(State.MID_TRANSPORT)

        elif self._state == State.MID_TRANSPORT:
            if self._position_reached(cfg.mid_transport_position):
                self._transition_to(State.PRE_PLACE)

        elif self._state == State.PRE_PLACE:
            if self._position_reached(cfg.pre_place_position):
                self._transition_to(State.PLACE_DESCEND)

        elif self._state == State.PLACE_DESCEND:
            if self._position_reached(cfg.place_position):
                self._transition_to(State.RELEASE)

        elif self._state == State.RELEASE:
            if self._time_in_state() >= cfg.release_settle_time:
                self._transition_to(State.PLACE_ASCEND)

        elif self._state == State.PLACE_ASCEND:
            if self._position_reached(cfg.pre_place_position):
                self._transition_to(State.RETURN_IDLE)

        elif self._state == State.RETURN_IDLE:
            if self._position_reached(cfg.idle_position):
                self._transition_to(State.IDLE)
                self.get_logger().info('Pick-place cycle complete. Waiting for next trigger.')

    def _publish_commands(self):
        """Publish target pose and gripper command."""
        # Target pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x = self._current_target_pos[0]
        pose_msg.pose.position.y = self._current_target_pos[1]
        pose_msg.pose.position.z = self._current_target_pos[2]
        pose_msg.pose.orientation.x = self._current_target_quat[0]
        pose_msg.pose.orientation.y = self._current_target_quat[1]
        pose_msg.pose.orientation.z = self._current_target_quat[2]
        pose_msg.pose.orientation.w = self._current_target_quat[3]
        self._target_pose_pub.publish(pose_msg)

        # Gripper command
        gripper_msg = Float64()
        gripper_msg.data = self._gripper_command
        self._gripper_pub.publish(gripper_msg)

    def _policy_loop(self):
        """Main policy loop - runs at policy_rate Hz."""
        with self._lock:
            # Wait for EE pose before doing anything
            if self._ee_pos is None:
                return

            # Initialize current target to EE position on first run
            if self._state == State.INIT and self._state_start_time is None:
                self._current_target_pos = self._ee_pos.copy()
                if self._ee_quat is not None:
                    self._current_target_quat = self._ee_quat.copy()
                self._state_start_time = self.get_clock().now()
                self.get_logger().info('EE pose received. Moving to idle position...')

            # Compute desired target for current state
            desired_pos, desired_quat, gripper = self._compute_desired_target()
            self._gripper_command = gripper

            # Update target with velocity limiting
            dt = 1.0 / self._config.policy_rate
            self._update_target_with_velocity_limit(desired_pos, desired_quat, dt)

            # Check for state transitions
            self._check_transitions()

            # Publish commands
            self._publish_commands()

            # Debug print at ~2 Hz
            self._debug_counter = getattr(self, '_debug_counter', 0) + 1
            if self._debug_counter % max(1, int(self._config.policy_rate / 2)) == 0:
                pos_err_vec = desired_pos - self._ee_pos
                pos_err_norm = np.linalg.norm(pos_err_vec)
                quat_err_vec = desired_quat - self._ee_quat
                quat_err_norm = np.linalg.norm(quat_err_vec)
                obj_str = (f"[{self._object_pos[0]:.4f}, {self._object_pos[1]:.4f}, {self._object_pos[2]:.4f}]"
                           if self._object_pos is not None else "None")
                # Green if under threshold, red if over
                GRN, RED, RST = '\033[32m', '\033[31m', '\033[0m'
                pos_color = GRN if pos_err_norm < self._config.position_threshold else RED
                quat_color = GRN if quat_err_norm < self._config.orientation_threshold else RED
                print(
                    f"\033[2J\033[H"  # clear screen
                    f"═══ Pick-Place Policy Debug ═══\n"
                    f"State:    {self._state.name}\n"
                    f"Gripper:  {'CLOSED' if self._gripper_command > 0.5 else 'OPEN'} ({self._gripper_command:.1f})\n"
                    f"\n"
                    f"EE pos:      [{self._ee_pos[0]:.4f}, {self._ee_pos[1]:.4f}, {self._ee_pos[2]:.4f}]\n"
                    f"EE quat:     [{self._ee_quat[0]:.4f}, {self._ee_quat[1]:.4f}, {self._ee_quat[2]:.4f}, {self._ee_quat[3]:.4f}]\n"
                    f"\n"
                    f"Desired pos: [{desired_pos[0]:.4f}, {desired_pos[1]:.4f}, {desired_pos[2]:.4f}]  (state goal)\n"
                    f"Target pos:  [{self._current_target_pos[0]:.4f}, {self._current_target_pos[1]:.4f}, {self._current_target_pos[2]:.4f}]  (vel-limited, sent to robot)\n"
                    f"Target quat: [{self._current_target_quat[0]:.4f}, {self._current_target_quat[1]:.4f}, {self._current_target_quat[2]:.4f}, {self._current_target_quat[3]:.4f}]\n"
                    f"\n"
                    f"Pos error:   [{pos_err_vec[0]:.4f}, {pos_err_vec[1]:.4f}, {pos_err_vec[2]:.4f}]  {pos_color}norm={pos_err_norm:.4f}{RST}  (thresh: {self._config.position_threshold})\n"
                    f"Quat error:  [{quat_err_vec[0]:.4f}, {quat_err_vec[1]:.4f}, {quat_err_vec[2]:.4f}, {quat_err_vec[3]:.4f}]  {quat_color}norm={quat_err_norm:.4f}{RST}  (thresh: {self._config.orientation_threshold})\n"
                    f"\n"
                    f"Object pos:  {obj_str}\n"
                    f"Detection fresh: {self._is_detection_fresh()}\n",
                    flush=True,
                )


def main(args=None):
    rclpy.init(args=args)
    node = PickPlacePolicy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
