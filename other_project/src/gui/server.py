"""
Viser-based GUI server.

Provides web-based visualization and control panels.
"""

import queue
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np

import viser
from viser.extras import ViserUrdf
import yourdfpy

from ..core.bus import MessageBus, Topics
from ..core.messages import UIState, ControllerState, SystemCommand, SystemCommandType, DesiredJoints
from ..core.config import Config


# URDF path
SCRIPT_DIR = Path(__file__).parent
URDF_PATH = SCRIPT_DIR / ".." / "assets" / "robots" / "kinova" / "urdf" / "gen3_2f85.urdf"


class GUIServer:
    """
    Viser-based web GUI for robot control and visualization.

    Subscribes to /ui/state for display updates.
    Publishes SystemCommand on button clicks.
    """

    def __init__(self, bus: MessageBus, config: Config, port: int = 8080):
        self.bus = bus
        self.config = config
        self.port = port

        # Viser server and URDF
        self.server: Optional[viser.ViserServer] = None
        self.viser_urdf: Optional[ViserUrdf] = None

        # GUI handles
        self._status_text = None
        self._state_text = None
        self._health_text = None
        self._connect_btn = None
        self._disconnect_btn = None

        self._joint_displays = []
        self._ee_displays: Dict[str, Any] = {}

        self._go_home_btn = None
        self._joint_control_btn = None
        self._joint_sliders = []

        self._gripper_slider = None

        self._diffik_btn = None
        self._target_displays: Dict[str, Any] = {}

        self._show_ee_frame_checkbox = None

        self._camera_toggle = None  # Camera on/off toggle
        self._camera_image = None  # Viser image display handle

        self._stop_btn = None

        # Scene elements
        self._ee_frame = None  # End-effector frame
        self._target_frame = None
        self._bounds_lines = []
        self._obstacle_meshes = []  # List of static obstacle visual elements
        self._dynamic_objects = {}  # Dict[name, mesh_handle] for dynamic objects
        self._ground_grid = None  # Ground grid reference

        # State subscriptions
        self._ui_queue: Optional[queue.Queue] = None
        self._objects_queue: Optional[queue.Queue] = None
        self._camera_image_queue: Optional[queue.Queue] = None

        # Callbacks (set by orchestrator)
        self._callbacks: Dict[str, Callable] = {}

        # Current state for UI logic
        self._current_state = ControllerState.DISCONNECTED
        self._previous_state = ControllerState.DISCONNECTED

        # Camera state
        self._camera_running = False
        self._camera_desired_state = False  # What the toggle is set to

    def setup(self) -> None:
        """Initialize Viser server and GUI panels."""
        # Create server
        self.server = viser.ViserServer(port=self.port)

        # Load URDF
        urdf = yourdfpy.URDF.load(
            str(URDF_PATH),
            load_meshes=True,
            build_scene_graph=True,
            load_collision_meshes=False,
        )
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=urdf,
            root_node_name="/robot",
        )

        # Add scene elements
        self.server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.01)
        # Initial grid at z=0 (will be updated when scene is loaded)
        # Using 'xy' plane for horizontal ground (z is vertical)
        self._ground_grid = self.server.scene.add_grid(
            "/grid",
            width=4.0,
            height=4.0,
            plane='xy',
            position=(0, 0, 0)
        )

        # End-effector frame (initially hidden, controlled by checkbox)
        self._ee_frame = self.server.scene.add_frame(
            "/ee_frame", axes_length=0.15, axes_radius=0.006
        )
        self._ee_frame.visible = False

        self._target_frame = self.server.scene.add_frame(
            "/target_ee", axes_length=0.1, axes_radius=0.005
        )
        self._target_frame.visible = False

        # Setup GUI panels
        self._setup_status_panel()
        self._setup_robot_state_panel()
        self._setup_joint_control_panel()
        self._setup_gripper_control_panel()
        self._setup_task_space_panel()
        self._setup_perception_panel()
        self._setup_stop_button()

        # Subscribe to UI state and scene objects
        self._ui_queue = self.bus.subscribe_queue(Topics.UI_STATE, maxsize=2)
        self._objects_queue = self.bus.subscribe_queue(Topics.SCENE_OBJECTS, maxsize=2)

        # Subscribe to camera images (realsense camera)
        camera_topic = Topics.PERCEPTION_IMAGE.replace("{camera_id}", "realsense")
        self._camera_image_queue = self.bus.subscribe_queue(camera_topic, maxsize=2)

        print(f"[GUI] Server running at http://localhost:{self.port}")

    def set_callback(self, name: str, callback: Callable) -> None:
        """Register a callback for GUI actions."""
        self._callbacks[name] = callback

    def set_camera_running(self, running: bool) -> None:
        """Update camera running state."""
        self._camera_running = running

        # Update toggle to reflect actual state (in case camera failed to start)
        if self._camera_toggle is not None:
            self._camera_toggle.value = running

        # Clear image display immediately when camera stops
        if not running and self._camera_image is not None:
            blank_image = np.zeros((240, 320, 3), dtype=np.uint8)
            self._camera_image.image = blank_image

    def _setup_status_panel(self) -> None:
        """Setup status and connection panel."""
        with self.server.gui.add_folder("Status"):
            self._state_text = self.server.gui.add_text(
                "State", initial_value="DISCONNECTED", disabled=True
            )
            self._status_text = self.server.gui.add_text(
                "Message", initial_value="Not connected", disabled=True
            )
            self._health_text = self.server.gui.add_text(
                "Connection", initial_value="---", disabled=True
            )

            self._connect_btn = self.server.gui.add_button("Connect to Robot")
            self._disconnect_btn = self.server.gui.add_button("Disconnect")
            self._disconnect_btn.visible = False

            @self._connect_btn.on_click
            def _(_):
                if "on_connect" in self._callbacks:
                    self._callbacks["on_connect"]()

            @self._disconnect_btn.on_click
            def _(_):
                if "on_disconnect" in self._callbacks:
                    self._callbacks["on_disconnect"]()

    def _setup_robot_state_panel(self) -> None:
        """Setup robot state display panel."""
        with self.server.gui.add_folder("Robot State"):
            for i in range(7):
                display = self.server.gui.add_number(
                    f"Joint {i+1} (deg)", initial_value=0.0, disabled=True
                )
                self._joint_displays.append(display)

        with self.server.gui.add_folder("End Effector"):
            self._show_ee_frame_checkbox = self.server.gui.add_checkbox(
                "Show EE Frame", initial_value=False
            )

            @self._show_ee_frame_checkbox.on_update
            def _(_):
                self._ee_frame.visible = self._show_ee_frame_checkbox.value

            self._ee_displays['x'] = self.server.gui.add_number(
                "X (m)", initial_value=0.0, disabled=True
            )
            self._ee_displays['y'] = self.server.gui.add_number(
                "Y (m)", initial_value=0.0, disabled=True
            )
            self._ee_displays['z'] = self.server.gui.add_number(
                "Z (m)", initial_value=0.0, disabled=True
            )

    def _setup_joint_control_panel(self) -> None:
        """Setup joint space control panel."""
        with self.server.gui.add_folder("Joint Space Control"):
            self._go_home_btn = self.server.gui.add_button("Go to Home")
            self._go_home_btn.disabled = True

            self._joint_control_btn = self.server.gui.add_button("Start Joint Control")
            self._joint_control_btn.disabled = True

            # Joint sliders
            joint_limits = [
                (-np.pi, np.pi),
                (-2.41, 2.41),
                (-np.pi, np.pi),
                (-2.66, 2.66),
                (-np.pi, np.pi),
                (-2.23, 2.23),
                (-np.pi, np.pi),
            ]

            for i, (lower, upper) in enumerate(joint_limits):
                slider = self.server.gui.add_slider(
                    f"J{i+1} Target (rad)",
                    min=lower,
                    max=upper,
                    step=0.01,
                    initial_value=0.0,
                    disabled=True
                )
                self._joint_sliders.append(slider)

                def make_slider_callback(joint_idx):
                    def callback(_):
                        if self._current_state == ControllerState.JOINT_CONTROL:
                            self._publish_slider_targets()
                    return callback

                slider.on_update(make_slider_callback(i))

            @self._go_home_btn.on_click
            def _(_):
                if "on_go_home" in self._callbacks:
                    self._callbacks["on_go_home"]()

            @self._joint_control_btn.on_click
            def _(_):
                if self._current_state == ControllerState.IDLE:
                    if "on_joint_control_start" in self._callbacks:
                        self._callbacks["on_joint_control_start"]()
                elif self._current_state == ControllerState.JOINT_CONTROL:
                    if "on_stop" in self._callbacks:
                        self._callbacks["on_stop"]()

    def _setup_gripper_control_panel(self) -> None:
        """Setup gripper control panel."""
        with self.server.gui.add_folder("Gripper Control"):
            self._gripper_slider = self.server.gui.add_slider(
                "Gripper Position",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.0,
                disabled=True
            )

            def gripper_callback(_):
                # Publish gripper command when slider changes
                if self._current_state in (ControllerState.IDLE, ControllerState.JOINT_CONTROL):
                    self._publish_gripper_target()

            self._gripper_slider.on_update(gripper_callback)

    def _setup_task_space_panel(self) -> None:
        """Setup task space control panel."""
        with self.server.gui.add_folder("Task Space Control"):
            self._diffik_btn = self.server.gui.add_button("Start Diff-IK")
            self._diffik_btn.disabled = True

            self.server.gui.add_markdown(
                """
**Keyboard Controls (when Diff-IK active):**
- Position: W/S (X), A/D (Y), Q/E (Z)
- Rotation: I/K (Rx), J/L (Ry), U/O (Rz)
"""
            )

            self._target_displays['x'] = self.server.gui.add_number(
                "Target X (m)", initial_value=0.0, disabled=True
            )
            self._target_displays['y'] = self.server.gui.add_number(
                "Target Y (m)", initial_value=0.0, disabled=True
            )
            self._target_displays['z'] = self.server.gui.add_number(
                "Target Z (m)", initial_value=0.0, disabled=True
            )

            @self._diffik_btn.on_click
            def _(_):
                if self._current_state == ControllerState.IDLE:
                    if "on_diffik_start" in self._callbacks:
                        self._callbacks["on_diffik_start"]()
                elif self._current_state in (ControllerState.DIFFIK_INIT, ControllerState.DIFFIK_ACTIVE):
                    if "on_stop" in self._callbacks:
                        self._callbacks["on_stop"]()

    def _setup_perception_panel(self) -> None:
        """Setup perception control panel."""
        with self.server.gui.add_folder("Perception"):
            # Camera on/off toggle
            self._camera_toggle = self.server.gui.add_checkbox(
                "Camera On/Off",
                initial_value=False
            )

            @self._camera_toggle.on_update
            def _(_):
                # Just record what the user wants
                self._camera_desired_state = self._camera_toggle.value

            # Add image display
            # Initial blank image (small placeholder)
            blank_image = np.zeros((240, 320, 3), dtype=np.uint8)
            self._camera_image = self.server.gui.add_image(
                image=blank_image,
                label="Camera Feed"
            )

    def _setup_stop_button(self) -> None:
        """Setup global stop button."""
        self.server.gui.add_markdown("---")
        self._stop_btn = self.server.gui.add_button("STOP")
        self._stop_btn.disabled = True

        @self._stop_btn.on_click
        def _(_):
            if "on_stop" in self._callbacks:
                self._callbacks["on_stop"]()

    def _publish_slider_targets(self) -> None:
        """Publish slider targets as desired joint positions."""
        values_rad = tuple(slider.value for slider in self._joint_sliders)

        from ..core.messages import DesiredJoints
        msg = DesiredJoints(positions=values_rad, source="sliders")
        self.bus.publish(Topics.CONTROL_DESIRED, msg)

    def _publish_gripper_target(self) -> None:
        """Publish gripper target position."""
        gripper_value = self._gripper_slider.value
        # Gripper value is 0-1, convert to MuJoCo control range 0-255
        gripper_ctrl = gripper_value * 255.0

        from ..core.messages import GripperCommand
        msg = GripperCommand(position=gripper_ctrl)
        self.bus.publish(Topics.GRIPPER_COMMAND, msg)

    def update(self) -> None:
        """Update GUI from latest state (call at 50Hz)."""
        # Get latest UI state
        ui_state: Optional[UIState] = None
        try:
            while True:
                ui_state = self._ui_queue.get_nowait()
        except queue.Empty:
            pass

        if ui_state is None:
            return

        # Track state transition
        self._previous_state = self._current_state
        self._current_state = ui_state.controller_state

        # Detect transition to JOINT_CONTROL
        just_entered_joint_control = (
            self._previous_state != ControllerState.JOINT_CONTROL and
            self._current_state == ControllerState.JOINT_CONTROL
        )

        # Update displays
        self._state_text.value = ui_state.controller_state.value.upper()
        self._status_text.value = ui_state.status_message

        if ui_state.controller_state == ControllerState.ERROR:
            self._health_text.value = "ERROR"
        elif ui_state.connection_healthy:
            self._health_text.value = "OK"
        else:
            self._health_text.value = "---"

        # Connection buttons
        is_disconnected = ui_state.controller_state == ControllerState.DISCONNECTED
        is_connecting = ui_state.controller_state == ControllerState.CONNECTING
        is_error = ui_state.controller_state == ControllerState.ERROR

        self._connect_btn.visible = is_disconnected or is_error
        self._connect_btn.disabled = is_connecting
        self._disconnect_btn.visible = not is_disconnected and not is_connecting and not is_error

        # Joint displays
        for i, deg in enumerate(ui_state.joint_positions_deg):
            self._joint_displays[i].value = round(float(deg), 2)

        # EE position
        self._ee_displays['x'].value = round(ui_state.ee_position[0], 4)
        self._ee_displays['y'].value = round(ui_state.ee_position[1], 4)
        self._ee_displays['z'].value = round(ui_state.ee_position[2], 4)

        # Control panel states
        is_idle = ui_state.controller_state == ControllerState.IDLE
        is_joint_control = ui_state.controller_state == ControllerState.JOINT_CONTROL
        is_going_home = ui_state.controller_state == ControllerState.GOING_HOME
        is_diffik = ui_state.controller_state in (ControllerState.DIFFIK_INIT, ControllerState.DIFFIK_ACTIVE)
        is_active = is_joint_control or is_going_home or is_diffik

        # Joint control panel
        self._go_home_btn.disabled = not is_idle
        self._joint_control_btn.disabled = not (is_idle or is_joint_control)
        self._joint_control_btn.name = "Stop Joint Control" if is_joint_control else "Start Joint Control"

        # Joint sliders
        for i, slider in enumerate(self._joint_sliders):
            slider.disabled = not is_joint_control
            # Update sliders when NOT in joint control OR when just entering joint control
            if not is_joint_control or just_entered_joint_control:
                slider.value = float(np.radians(ui_state.joint_positions_deg[i]))

        # Gripper control
        can_control_gripper = is_idle or is_joint_control
        self._gripper_slider.disabled = not can_control_gripper

        # Task space panel
        self._diffik_btn.disabled = not (is_idle or is_diffik)
        self._diffik_btn.name = "Stop Diff-IK" if is_diffik else "Start Diff-IK"

        # Target pose
        self._target_displays['x'].value = round(ui_state.target_position[0], 4)
        self._target_displays['y'].value = round(ui_state.target_position[1], 4)
        self._target_displays['z'].value = round(ui_state.target_position[2], 4)

        # Stop button
        self._stop_btn.disabled = not is_active

        # Handle camera toggle state changes
        if self._camera_desired_state != self._camera_running:
            if "on_camera_toggle" in self._callbacks:
                self._callbacks["on_camera_toggle"]()

        # Update visualization
        self._update_visualization(ui_state)

        # Update dynamic objects
        self._update_dynamic_objects()

        # Update camera image
        self._update_camera_image()

    def _update_visualization(self, ui_state: UIState) -> None:
        """Update 3D visualization."""
        # Update robot mesh
        q_rad = np.radians(ui_state.joint_positions_deg)
        if len(q_rad) == 7:
            # Add actual gripper position from UI state
            q_full = np.concatenate([q_rad, [ui_state.gripper_position]])
        else:
            q_full = q_rad[:8]
        self.viser_urdf.update_cfg(q_full)

        # Update end-effector frame (only if checkbox is enabled)
        if self._show_ee_frame_checkbox.value:
            self._ee_frame.position = ui_state.ee_position
            # Convert quaternion from (x, y, z, w) to (w, x, y, z) for viser
            quat_xyzw = ui_state.ee_orientation
            quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
            self._ee_frame.wxyz = quat_wxyz

        # Update target frame
        if ui_state.controller_state == ControllerState.DIFFIK_ACTIVE:
            self._target_frame.visible = True
            self._target_frame.position = ui_state.target_position
        else:
            self._target_frame.visible = False

    def set_ground_height(self, height: float) -> None:
        """
        Set the ground plane height.

        Args:
            height: Ground height in meters (z-coordinate)
        """
        print(f"[GUI] Setting ground height to {height}m")
        if self._ground_grid is not None:
            self._ground_grid.position = (0, 0, height)

    def set_position_bounds(self, pos_min: np.ndarray, pos_max: np.ndarray) -> None:
        """Display position bounds as wireframe box."""
        print(f"[GUI] Setting position bounds: min={pos_min}, max={pos_max}")

        # Remove existing
        for line in self._bounds_lines:
            line.remove()
        self._bounds_lines = []

        # Define corners
        corners = np.array([
            [pos_min[0], pos_min[1], pos_min[2]],
            [pos_max[0], pos_min[1], pos_min[2]],
            [pos_max[0], pos_max[1], pos_min[2]],
            [pos_min[0], pos_max[1], pos_min[2]],
            [pos_min[0], pos_min[1], pos_max[2]],
            [pos_max[0], pos_min[1], pos_max[2]],
            [pos_max[0], pos_max[1], pos_max[2]],
            [pos_min[0], pos_max[1], pos_max[2]],
        ])

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
        ]

        print(f"[GUI] Creating {len(edges)} boundary lines")
        for i, (start, end) in enumerate(edges):
            line = self.server.scene.add_spline_catmull_rom(
                f"/bounds/edge_{i}",
                positions=np.array([corners[start], corners[end]]),
                color=(255, 0, 0),
                line_width=5.0,
            )
            self._bounds_lines.append(line)

        print(f"[GUI] Position bounds visualization created with {len(self._bounds_lines)} lines")

    def load_scene_obstacles(self, obstacles: list) -> None:
        """
        Load obstacles into the Viser scene.

        Args:
            obstacles: List of obstacle dicts with name, type, position, orientation, size, color
        """
        # Clear existing obstacles
        for mesh in self._obstacle_meshes:
            mesh.remove()
        self._obstacle_meshes = []

        print(f"[GUI] Loading {len(obstacles)} obstacles into scene")

        for obs in obstacles:
            name = obs["name"]
            obs_type = obs["type"]
            position = obs["position"]
            orientation = obs["orientation"]  # [x, y, z, w]
            color = obs.get("color", (0.8, 0.8, 0.8))

            # For cuboids, generate inline with scene config colors (don't use URDF)
            if obs_type == "cuboid":
                # Add box mesh with scene config color
                size = obs.get("size", [0.1, 0.1, 0.1])

                # Convert quaternion from [x, y, z, w] to wxyz for Viser
                quat_xyzw = orientation
                quat_wxyz_viser = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                mesh = self.server.scene.add_box(
                    f"/obstacles/{name}",
                    dimensions=tuple(size),
                    position=tuple(position),
                    wxyz=tuple(quat_wxyz_viser),
                    color=color,
                )
                self._obstacle_meshes.append(mesh)
                print(f"[GUI]   - Added cuboid '{name}' at {position} with size {size} and color {color}")
                continue

            # Check if object has URDF (for mesh loading - e.g., table)
            urdf_path = obs.get("urdf_path")

            if urdf_path:
                # Load URDF with meshes
                import yourdfpy
                from scipy.spatial.transform import Rotation as R

                # First create a transform frame at the desired pose
                quat_xyzw = orientation
                rot = R.from_quat(quat_xyzw)  # [x, y, z, w]

                transform_frame = self.server.scene.add_frame(
                    f"/obstacles/{name}",
                    wxyz=rot.as_quat()[[3, 0, 1, 2]],  # Convert to [w, x, y, z]
                    position=tuple(position),
                    show_axes=False,
                )

                # Load URDF
                urdf_obj = yourdfpy.URDF.load(
                    urdf_path,
                    load_meshes=True,
                    build_scene_graph=True,
                    load_collision_meshes=False,
                )

                # Create ViserUrdf as child of the transform frame
                from viser.extras import ViserUrdf
                urdf_handle = ViserUrdf(
                    self.server,
                    urdf_or_path=urdf_obj,
                    root_node_name=f"/obstacles/{name}/mesh",  # Child of transform frame
                )

                # Update configuration (empty for non-articulated objects)
                if len(urdf_obj.actuated_joints) > 0:
                    urdf_handle.update_cfg(np.zeros(len(urdf_obj.actuated_joints)))

                self._obstacle_meshes.append(transform_frame)
                self._obstacle_meshes.append(urdf_handle)
                print(f"[GUI]   - Added URDF '{name}' from {urdf_path} at {position}")
            else:
                print(f"[GUI]   - Warning: Unknown obstacle type '{obs_type}' (no URDF or mesh support) for '{name}'")

        print(f"[GUI] Loaded {len(self._obstacle_meshes)} obstacles")

    def load_dynamic_objects(self, objects: list) -> None:
        """
        Load dynamic objects into the Viser scene.

        Args:
            objects: List of object dicts with name, type, position, orientation, size, color
        """
        # Clear existing dynamic objects
        for mesh in self._dynamic_objects.values():
            mesh.remove()
        self._dynamic_objects = {}

        print(f"[GUI] Loading {len(objects)} dynamic objects into scene")

        for obj in objects:
            name = obj["name"]
            obj_type = obj["type"]
            position = obj["position"]
            orientation = obj["orientation"]  # [x, y, z, w]
            color = obj.get("color", (0.5, 0.5, 0.5))

            if obj_type == "cuboid":
                # Add box mesh
                size = obj.get("size", [0.05, 0.05, 0.05])

                # Convert quaternion from [x, y, z, w] to wxyz for Viser
                from scipy.spatial.transform import Rotation as R
                quat_xyzw = orientation
                rot = R.from_quat(quat_xyzw)  # scipy expects [x, y, z, w]
                quat_wxyz_viser = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                mesh = self.server.scene.add_box(
                    f"/dynamic_objects/{name}",
                    dimensions=tuple(size),
                    position=tuple(position),
                    wxyz=tuple(quat_wxyz_viser),
                    color=color,
                )
                self._dynamic_objects[name] = mesh
                print(f"[GUI]   - Added dynamic cuboid '{name}' at {position} with size {size}")
            else:
                print(f"[GUI]   - Warning: Unknown object type '{obj_type}' for '{name}'")

        print(f"[GUI] Loaded {len(self._dynamic_objects)} dynamic objects")

    def _update_dynamic_objects(self) -> None:
        """Update dynamic object poses from scene objects topic."""
        # Get latest scene objects state
        scene_objects = None
        try:
            while True:
                scene_objects = self._objects_queue.get_nowait()
        except queue.Empty:
            pass

        if scene_objects is None:
            return

        # Update poses
        for name, obj_state in scene_objects.objects.items():
            if name in self._dynamic_objects:
                mesh = self._dynamic_objects[name]

                # Update position
                mesh.position = obj_state.position

                # Update orientation (convert [x, y, z, w] to wxyz for Viser)
                quat_xyzw = obj_state.orientation
                quat_wxyz_viser = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
                mesh.wxyz = quat_wxyz_viser

    def _update_camera_image(self) -> None:
        """Update camera image display from perception topic."""
        # Get latest camera image
        camera_msg = None
        try:
            while True:
                camera_msg = self._camera_image_queue.get_nowait()
        except queue.Empty:
            pass

        if camera_msg is None:
            return

        # Convert image message to numpy array
        import numpy as np
        width = camera_msg.width
        height = camera_msg.height
        channels = camera_msg.channels

        # Decode image data
        image_array = np.frombuffer(camera_msg.data, dtype=np.uint8)
        image = image_array.reshape((height, width, channels))

        # Update the image display
        if self._camera_image is not None:
            self._camera_image.image = image

    def teardown(self) -> None:
        """Cleanup."""
        # Remove obstacles
        for mesh in self._obstacle_meshes:
            mesh.remove()
        self._obstacle_meshes = []

        # Remove dynamic objects
        for mesh in self._dynamic_objects.values():
            mesh.remove()
        self._dynamic_objects = {}
