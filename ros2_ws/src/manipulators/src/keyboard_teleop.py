"""
Keyboard teleop node.

Reads keyboard input and publishes incremental EE pose targets.

Keys:
  W/S  - X forward/back
  A/D  - Y left/right
  Q/E  - Z up/down
  I/K  - pitch +/-
  J/L  - yaw +/-
  U/O  - roll +/-
  G    - toggle gripper open/close
  ESC  - quit
"""

import sys
import tty
import termios
import select
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

from .utility import quat_to_matrix, matrix_to_quat


USAGE = """
Keyboard Teleop — Kinova Gen3
──────────────────────────────
  W/S : X forward/back
  A/D : Y left/right
  Q/E : Z up/down
  I/K : pitch +/-
  J/L : yaw +/-
  U/O : roll +/-
  G   : toggle gripper
  ESC : quit
──────────────────────────────
"""

# Map key -> (axis_index, sign)
# Position: indices 0-2 (x,y,z)  Rotation: indices 3-5 (roll,pitch,yaw)
KEY_MAP = {
    'w': (0, +1), 's': (0, -1),   # X
    'a': (1, +1), 'd': (1, -1),   # Y
    'q': (2, +1), 'e': (2, -1),   # Z
    'u': (3, +1), 'o': (3, -1),   # roll
    'i': (4, +1), 'k': (4, -1),   # pitch
    'j': (5, +1), 'l': (5, -1),   # yaw
}


def _get_key(timeout=0.1):
    """Read a single keypress (non-blocking with timeout)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def _rotation_matrix(axis: int, angle: float) -> np.ndarray:
    """Small rotation matrix around axis 0=X, 1=Y, 2=Z."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    if axis == 0:  # roll (X)
        R[1, 1] = c; R[1, 2] = -s
        R[2, 1] = s; R[2, 2] = c
    elif axis == 1:  # pitch (Y)
        R[0, 0] = c; R[0, 2] = s
        R[2, 0] = -s; R[2, 2] = c
    elif axis == 2:  # yaw (Z)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
    return R


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.declare_parameter('linear_step', 0.005)
        self.declare_parameter('angular_step', 0.05)
        self.declare_parameter('publish_rate', 30.0)

        self.linear_step = self.get_parameter('linear_step').value
        self.angular_step = self.get_parameter('angular_step').value
        self.publish_rate = self.get_parameter('publish_rate').value

        self.pose_pub = self.create_publisher(PoseStamped, 'target_pose', 1)
        self.gripper_pub = self.create_publisher(Float64, 'gripper_command', 1)

        # Wait for initial EE pose
        self._target_pos = None
        self._target_rot = None
        self._gripper_open = True

        self.create_subscription(PoseStamped, 'ee_pose', self._on_ee_pose, 1)

    def _on_ee_pose(self, msg: PoseStamped):
        """Capture initial EE pose (only used once to initialize target)."""
        if self._target_pos is not None:
            return
        self._target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self._target_rot = quat_to_matrix(np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]))
        self.get_logger().info("Got initial EE pose. Keyboard active.")

    def run(self):
        """Main loop: read keys, update target, publish."""
        print(USAGE)
        self.get_logger().info("Waiting for initial EE pose from control node ...")

        rate_dt = 1.0 / self.publish_rate

        while rclpy.ok():
            # Spin once to process ee_pose subscription
            rclpy.spin_once(self, timeout_sec=0.01)

            if self._target_pos is None:
                continue

            key = _get_key(timeout=rate_dt)

            if key == '\x1b':  # ESC
                self.get_logger().info("ESC pressed, exiting.")
                break

            if key and key.lower() == 'g':
                self._gripper_open = not self._gripper_open
                msg = Float64()
                msg.data = 0.0 if self._gripper_open else 1.0
                self.gripper_pub.publish(msg)
                status = "OPEN" if self._gripper_open else "CLOSED"
                self.get_logger().info(f"Gripper: {status}")
                continue

            if key and key.lower() in KEY_MAP:
                axis, sign = KEY_MAP[key.lower()]
                if axis < 3:
                    # Position increment
                    delta = np.zeros(3)
                    delta[axis] = sign * self.linear_step
                    self._target_pos += delta
                else:
                    # Rotation increment
                    rot_axis = axis - 3
                    dR = _rotation_matrix(rot_axis, sign * self.angular_step)
                    self._target_rot = dR @ self._target_rot

            # Always publish current target (even if no key pressed)
            self._publish_target()

    def _publish_target(self):
        quat = matrix_to_quat(self._target_rot)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(self._target_pos[0])
        msg.pose.position.y = float(self._target_pos[1])
        msg.pose.position.z = float(self._target_pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
