"""
Interactive camera-to-robot extrinsic calibration node.

Detects a chessboard in the camera feed, lets the user adjust the
board-to-robot transform via trackbars, and visualises coordinate axes
at the chessboard origin, robot base, and robot end-effector.

Press SPACE to save the computed extrinsics.  Press ESC to quit.
"""

import tkinter as tk

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from .utility import quat_to_matrix


# ── Helpers ──────────────────────────────────────────────────────────────

def _rotation_matrix_from_rpy(roll_deg, pitch_deg, yaw_deg):
    """RPY (degrees) → 3×3 rotation matrix (Rz Ry Rx)."""
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _project_origin(T_cam_frame, cam_mtx, cam_dist):
    """Return the 2-D pixel of a frame's origin (for label placement)."""
    pt = T_cam_frame[:3, 3].reshape(3, 1).astype(np.float64)
    px, _ = cv2.projectPoints(pt, np.zeros(3), np.zeros(3), cam_mtx, cam_dist)
    return int(px[0, 0, 0]), int(px[0, 0, 1])


_WIN = "Extrinsic Calibration"


# ── Node ─────────────────────────────────────────────────────────────────

class ExtrinsicCalibrationNode(Node):
    def __init__(self):
        super().__init__("extrinsic_calibration_node")

        # -- Parameters ------------------------------------------------
        self.declare_parameter("camera_yaml", "")
        self.declare_parameter("board_size", [5, 3])
        self.declare_parameter("square_size_mm", 42.1)
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("default_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("default_rpy", [0.0, 0.0, 0.0])

        camera_yaml = self.get_parameter("camera_yaml").value
        board_size = list(self.get_parameter("board_size").value)
        square_size_mm = self.get_parameter("square_size_mm").value
        image_topic = self.get_parameter("image_topic").value
        default_xyz = list(self.get_parameter("default_xyz").value)
        default_rpy = list(self.get_parameter("default_rpy").value)

        # -- Load intrinsics -------------------------------------------
        try:
            with open(camera_yaml, "r") as f:
                cam_data = yaml.safe_load(f)
            intr = cam_data["intrinsics"]
            self.cam_mtx = np.array(intr["camera_matrix"], dtype=np.float64).reshape(3, 3)
            self.cam_dist = np.array(intr["dist_coeffs"], dtype=np.float64)
        except Exception as e:
            self.get_logger().fatal(f"Cannot load intrinsics from {camera_yaml}: {e}")
            raise SystemExit(1)

        self.output_file = camera_yaml
        self.board_size = tuple(board_size)
        self.square_size_m = square_size_mm / 1000.0

        # Chessboard 3-D object points
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = (
            np.mgrid[0 : self.board_size[0], 0 : self.board_size[1]]
            .T.reshape(-1, 2)
        )
        self.objp *= self.square_size_m
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        )

        # -- State -----------------------------------------------------
        self.bridge = CvBridge()
        self._latest_frame = None
        self._ee_pos = None   # (3,)
        self._ee_rot = None   # (3,3)
        self._last_T_cam_board = None   # cached PnP result for when board is lost
        self._last_corners = None       # cached corners for saving
        self._board_locked = False      # L key toggles: skip detection, use cached

        # -- Subscriptions ---------------------------------------------
        self.create_subscription(Image, image_topic, self._on_image, 1)
        self.create_subscription(PoseStamped, "ee_pose", self._on_ee_pose, 1)

        self.get_logger().info(
            f"Subscribing to {image_topic} and /ee_pose. "
            "Adjust trackbars, press SPACE to save, ESC to quit."
        )

        # -- GUI -------------------------------------------------------
        cv2.namedWindow(_WIN)
        # Orientation trackbars (degrees)
        cv2.createTrackbar("Roll  (deg)", _WIN, int(default_rpy[0]) % 361, 360, lambda _: None)
        cv2.createTrackbar("Pitch (deg)", _WIN, int(default_rpy[1]) % 361, 360, lambda _: None)
        cv2.createTrackbar("Yaw   (deg)", _WIN, int(default_rpy[2]) % 361, 360, lambda _: None)

        # Tkinter panel for XYZ text entry
        self._tk_root = tk.Tk()
        self._tk_root.title("Board → Robot Transform")
        self._tk_root.attributes("-topmost", True)
        self._xyz_vars = []
        for i, (label, val) in enumerate([
            ("X (m)", default_xyz[0]),
            ("Y (m)", default_xyz[1]),
            ("Z (m)", default_xyz[2]),
        ]):
            tk.Label(self._tk_root, text=label, font=("monospace", 12)).grid(
                row=i, column=0, padx=5, pady=3, sticky="e"
            )
            var = tk.StringVar(value=f"{val:.4f}")
            tk.Entry(
                self._tk_root, textvariable=var, width=12, font=("monospace", 12)
            ).grid(row=i, column=1, padx=5, pady=3)
            self._xyz_vars.append(var)

    # ── Callbacks ─────────────────────────────────────────────────────

    def _on_image(self, msg: Image):
        self._latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _on_ee_pose(self, msg: PoseStamped):
        self._ee_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        quat_xyzw = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        self._ee_rot = quat_to_matrix(quat_xyzw)

    # ── Input helpers ────────────────────────────────────────────────

    def _read_inputs(self):
        """Return (xyz_m, rpy_deg) from Tkinter entries + trackbars."""
        try:
            x = float(self._xyz_vars[0].get())
            y = float(self._xyz_vars[1].get())
            z = float(self._xyz_vars[2].get())
        except ValueError:
            x, y, z = 0.0, 0.0, 0.0
        roll = cv2.getTrackbarPos("Roll  (deg)", _WIN)
        pitch = cv2.getTrackbarPos("Pitch (deg)", _WIN)
        yaw = cv2.getTrackbarPos("Yaw   (deg)", _WIN)
        return np.array([x, y, z]), np.array([roll, pitch, yaw], dtype=float)

    def _build_T_board_robot(self, xyz, rpy_deg):
        T = np.eye(4)
        T[:3, :3] = _rotation_matrix_from_rpy(*rpy_deg)
        T[:3, 3] = xyz
        return T

    # ── Drawing helpers ──────────────────────────────────────────────

    @staticmethod
    def _draw_labeled_axes(frame, cam_mtx, cam_dist, T_cam_frame,
                           length, label, color):
        """Draw coordinate axes + a text label for a given frame."""
        rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
        tvec = T_cam_frame[:3, 3].astype(np.float64)
        cv2.drawFrameAxes(frame, cam_mtx, cam_dist, rvec, tvec, length)
        try:
            px, py = _project_origin(T_cam_frame, cam_mtx, cam_dist)
            cv2.putText(frame, label, (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass

    # ── Main loop ────────────────────────────────────────────────────

    def run(self):
        saved = False
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)

            if self._latest_frame is None:
                continue

            frame = self._latest_frame.copy()

            try:
                self._tk_root.update()
            except tk.TclError:
                break  # Tkinter window closed

            xyz, rpy_deg = self._read_inputs()
            T_board_robot = self._build_T_board_robot(xyz, rpy_deg)

            # Resolve T_cam_board: live detection, locked, or cached fallback
            T_cam_board = None
            board_live = False

            if self._board_locked and self._last_T_cam_board is not None:
                # Locked — skip detection entirely, use frozen transform
                T_cam_board = self._last_T_cam_board
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, self.board_size, None
                )

                if ret:
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), self.criteria
                    )
                    cv2.drawChessboardCorners(frame, self.board_size, corners, ret)

                    ok, rvec_cb, tvec_cb = cv2.solvePnP(
                        self.objp, corners, self.cam_mtx, self.cam_dist
                    )
                    if ok:
                        R_cb, _ = cv2.Rodrigues(rvec_cb)
                        T_cam_board = np.eye(4)
                        T_cam_board[:3, :3] = R_cb
                        T_cam_board[:3, 3] = tvec_cb.flatten()
                        # Cache for when the board disappears
                        self._last_T_cam_board = T_cam_board.copy()
                        self._last_corners = corners.copy()
                        board_live = True
                elif self._last_T_cam_board is not None:
                    # Board not visible — use last known transform
                    T_cam_board = self._last_T_cam_board

            # Draw axes if we have a board transform (live or cached)
            if T_cam_board is not None:
                # 1. Axes at chessboard origin
                self._draw_labeled_axes(
                    frame, self.cam_mtx, self.cam_dist,
                    T_cam_board,
                    length=4 * self.square_size_m,
                    label="Board", color=(0, 255, 0),
                )

                # T_cam_robot = T_cam_board @ T_board_robot
                T_cam_robot = T_cam_board @ T_board_robot

                # 2. Axes at robot base origin
                self._draw_labeled_axes(
                    frame, self.cam_mtx, self.cam_dist,
                    T_cam_robot,
                    length=0.1,
                    label="Robot Base", color=(255, 180, 0),
                )

                # 3. Axes at robot end-effector
                if self._ee_pos is not None:
                    T_robot_ee = np.eye(4)
                    T_robot_ee[:3, :3] = self._ee_rot
                    T_robot_ee[:3, 3] = self._ee_pos
                    T_cam_ee = T_cam_robot @ T_robot_ee
                    self._draw_labeled_axes(
                        frame, self.cam_mtx, self.cam_dist,
                        T_cam_ee,
                        length=0.05,
                        label="EE", color=(0, 0, 255),
                    )

                # Text overlay
                cv2.putText(
                    frame,
                    f"XYZ: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                )
                cv2.putText(
                    frame,
                    f"RPY: [{rpy_deg[0]:.0f}, {rpy_deg[1]:.0f}, {rpy_deg[2]:.0f}] deg",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                )

                if self._board_locked:
                    status = "LOCKED (L=unlock)  |  SPACE=save  ESC=quit"
                    status_color = (0, 0, 255)
                elif board_live:
                    status = "Chessboard LIVE (L=lock)  |  SPACE=save  ESC=quit"
                    status_color = (0, 255, 0)
                else:
                    status = "Using cached pose (L=lock)  |  SPACE=save  ESC=quit"
                    status_color = (0, 200, 255)
                cv2.putText(
                    frame, status,
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1,
                )
            else:
                cv2.putText(
                    frame, "No chessboard detected (show board once to cache)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            cv2.imshow(_WIN, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('l') or key == ord('L'):
                if self._last_T_cam_board is not None:
                    self._board_locked = not self._board_locked
                    state = "LOCKED" if self._board_locked else "UNLOCKED"
                    self.get_logger().info(f"Board transform {state}")
            elif key == 32 and self._last_corners is not None:  # SPACE
                self._save(self._last_corners, T_board_robot)
                saved = True
                break

        cv2.destroyAllWindows()
        try:
            self._tk_root.destroy()
        except tk.TclError:
            pass
        if saved:
            self.get_logger().info("Calibration saved. Exiting.")
        else:
            self.get_logger().info("Exited without saving.")

    # ── Save ─────────────────────────────────────────────────────────

    def _save(self, corners, T_board_robot):
        """Compute final camera-to-robot transform and write to YAML."""
        ret, rvec_cb, tvec_cb = cv2.solvePnP(
            self.objp, corners, self.cam_mtx, self.cam_dist
        )
        if not ret:
            self.get_logger().error("solvePnP failed — nothing saved.")
            return

        R_cb, _ = cv2.Rodrigues(rvec_cb)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_cb
        T_cam_board[:3, 3] = tvec_cb.flatten()

        T_cam_robot = T_cam_board @ T_board_robot

        rvec_cr, _ = cv2.Rodrigues(T_cam_robot[:3, :3])
        tvec_cr = T_cam_robot[:3, 3]

        self.get_logger().info(f"Camera-to-robot transform:")
        self.get_logger().info(f"  rvec: {rvec_cr.flatten()}")
        self.get_logger().info(f"  tvec: {tvec_cr.flatten()}")

        # Merge with existing YAML
        data = {}
        try:
            with open(self.output_file, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass

        data["extrinsics"] = {
            "rvec": rvec_cr.flatten().tolist(),
            "tvec": tvec_cr.flatten().tolist(),
            "T_cam_robot": T_cam_robot.flatten().tolist(),
        }

        with open(self.output_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        self.get_logger().info(f"Saved extrinsics to {self.output_file}")


# ── Entry point ──────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicCalibrationNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
