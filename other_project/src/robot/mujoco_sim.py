"""
MuJoCo simulation backend for Kinova Gen3.

Implements the HardwareInterface to provide simulation as a drop-in replacement
for real hardware.
"""

import time
import numpy as np
from typing import Optional
from pathlib import Path
import threading
import multiprocessing as mp

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("[MuJoCo] Warning: mujoco package not found. Install with: pip install mujoco")

from .hardware_base import HardwareInterface, RobotFeedback
from ..core.bus import MessageBus
from ..core.scene_config import SceneConfig


def _viewer_process_main(model_path: str, stop_event: mp.Event, qpos_array: mp.Array):
    """
    Viewer process main function (runs in separate process).

    This runs in a completely separate process to avoid OpenGL context
    conflicts with the Viser GUI server in the main process.

    Args:
        model_path: Path to MuJoCo XML model
        stop_event: Event to signal viewer shutdown
        qpos_array: Shared memory array for joint positions (updated by main process)
    """
    try:
        # Import mujoco in this process
        import mujoco
        import mujoco.viewer
        import numpy as np

        # Load model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Launch viewer in this process (this process's main thread)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"[MuJoCoViewer] Window opened")

            # Keep viewer alive and sync at 60 FPS
            while not stop_event.is_set():
                # Update joint positions from shared memory
                with qpos_array.get_lock():
                    qpos_shared = np.frombuffer(qpos_array.get_obj(), dtype=np.float64)
                    data.qpos[:len(qpos_shared)] = qpos_shared

                # Forward kinematics to update visualization
                mujoco.mj_forward(model, data)

                # Sync viewer
                viewer.sync()
                time.sleep(1.0 / 60.0)

        print(f"[MuJoCoViewer] Closed")

    except Exception as e:
        print(f"[MuJoCoViewer] Error: {e}")
        import traceback
        traceback.print_exc()


class MuJoCoSimulator(HardwareInterface):
    """
    MuJoCo simulation backend that mimics Kinova hardware interface.

    Provides the same interface as KinovaHardware but runs in simulation.
    Supports both position and torque control modes.
    """

    def __init__(self, model_path: str, render: bool = True, home_joints_rad: Optional[np.ndarray] = None,
                 bus: Optional[MessageBus] = None, scene_config: Optional[SceneConfig] = None):
        """
        Initialize MuJoCo simulator.

        Args:
            model_path: Path to MuJoCo XML model file
            render: Enable visualization window
            home_joints_rad: Home joint configuration (7,) in radians
            bus: Message bus for publishing object states (optional)
            scene_config: Scene configuration to identify dynamic objects (optional)
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo not available. Install with: pip install mujoco")

        self.model_path = model_path
        self.render_enabled = render
        self.home_joints_rad = home_joints_rad
        self.bus = bus
        self.scene_config = scene_config

        # MuJoCo components
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer = None

        # Simulation state
        self._time = 0.0
        self._dt = 0.001  # 1kHz to match hardware (1ms timestep)
        self._in_torque_mode = False
        self._is_ready = False

        # Control state
        self._last_positions_deg = np.zeros(7)
        self._frame_id = 0

        # Real-time tracking
        self._last_step_time = None

        # Viewer process (separate process to avoid OpenGL conflicts)
        self._viewer_process = None
        self._viewer_running = False
        self._viewer_stop_event = None
        self._viewer_qpos_shared = None  # Shared memory for joint positions

        # Simulation thread (runs physics independently)
        self._sim_thread = None
        self._sim_running = False

        # Thread synchronization for safe access to model/data
        self._data_lock = threading.Lock()

        # Object state publishing (for dynamic objects)
        self._dynamic_object_names = []
        self._object_joint_ids = {}  # Map object name to freejoint qpos indices
        self._last_object_publish_time = 0.0
        self._object_publish_rate = 0.05  # Publish at 20Hz (every 50ms)

        # Extract dynamic object names from scene config
        if self.scene_config:
            self._dynamic_object_names = [obj.name for obj in self.scene_config.objects if not obj.static]
            print(f"[MuJoCoSim] Dynamic objects to track: {self._dynamic_object_names}")

    def connect(self) -> bool:
        """
        Load MuJoCo model and initialize simulation.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Resolve path
            model_file = Path(self.model_path)
            if not model_file.exists():
                print(f"[MuJoCoSim] Model file not found: {self.model_path}")
                return False

            print(f"[MuJoCoSim] Loading model: {self.model_path}")

            # Load model
            self.model = mujoco.MjModel.from_xml_path(str(model_file))
            self.data = mujoco.MjData(self.model)

            # Set timestep to match hardware loop (1kHz = 0.001s)
            self.model.opt.timestep = self._dt

            # Verify model has correct number of joints
            if self.model.nq < 7:
                print(f"[MuJoCoSim] Error: Model has only {self.model.nq} joints, need 7")
                return False

            # Set initial position programmatically (not using keyframes)
            # Home position for robot (7 arm joints)
            home_qpos = np.array([0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633])

            if self.home_joints_rad is not None:
                home_qpos = self.home_joints_rad[:7]

            # Set robot joint positions (first 7 joints)
            self.data.qpos[:7] = home_qpos
            self.data.ctrl[:7] = home_qpos

            # Gripper joints (8 joints starting at index 7) - set to closed position
            if self.model.nq >= 15:
                self.data.qpos[7:15] = 0  # All gripper joints at 0

            # Dynamic objects already have their initial poses set from body definitions
            # MuJoCo automatically initializes freejoint qpos from body pos/quat

            mujoco.mj_forward(self.model, self.data)
            print(f"[MuJoCoSim] Initialized to home position: {np.degrees(home_qpos)}")
            print(f"[MuJoCoSim] Total DOF (nq): {self.model.nq}")

            self._last_positions_deg = np.degrees(self.data.qpos[:7].copy())
            self._is_ready = True
            self._last_step_time = time.time()

            # Map dynamic object names to their freejoint qpos indices
            self._map_object_joints()

            # Start simulation thread (physics runs independently at 1kHz)
            self._start_simulation()

            print(f"[MuJoCoSim] Connected successfully")
            print(f"[MuJoCoSim] Model: {self.model.nu} actuators, {self.model.nq} joints")
            print(f"[MuJoCoSim] Timestep: {self.model.opt.timestep*1000:.2f}ms (real-time enabled)")
            return True

        except Exception as e:
            print(f"[MuJoCoSim] Connection error: {e}")
            self.disconnect()
            return False

    def _map_object_joints(self):
        """Map dynamic object names to their freejoint qpos indices."""
        if not self._dynamic_object_names or self.model is None:
            return

        for obj_name in self._dynamic_object_names:
            # Find the body with this name
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id == -1:
                print(f"[MuJoCoSim] Warning: Dynamic object '{obj_name}' not found in model")
                continue

            # Get the joint for this body (should be a freejoint)
            # Find first joint attached to this body
            joint_id = -1
            for jnt_id in range(self.model.njnt):
                if self.model.jnt_bodyid[jnt_id] == body_id:
                    joint_id = jnt_id
                    break

            if joint_id == -1:
                print(f"[MuJoCoSim] Warning: No joint found for object '{obj_name}'")
                continue

            # Get qpos address for this joint (freejoint has 7 values: 3 pos + 4 quat)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self._object_joint_ids[obj_name] = qpos_addr

            print(f"[MuJoCoSim] Mapped object '{obj_name}' to qpos index {qpos_addr}")

    def _publish_object_states(self):
        """Publish dynamic object states to the message bus."""
        if not self.bus or not self._object_joint_ids or self.data is None:
            return

        # Check if it's time to publish (rate limiting to 20Hz)
        current_time = time.time()
        if current_time - self._last_object_publish_time < self._object_publish_rate:
            return

        self._last_object_publish_time = current_time

        # Create object states
        from ..core.messages import ObjectState, SceneObjects

        objects = {}
        with self._data_lock:
            for obj_name, qpos_idx in self._object_joint_ids.items():
                # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
                # Freejoint qvel: [vx, vy, vz, wx, wy, wz]
                pos = self.data.qpos[qpos_idx:qpos_idx+3]
                quat_wxyz = self.data.qpos[qpos_idx+3:qpos_idx+7]  # MuJoCo stores as [w, x, y, z]

                # Get velocities (freejoint qvel is 6D: 3 linear + 3 angular)
                # Find qvel address for this joint
                joint_id = -1
                for jnt_id in range(self.model.njnt):
                    if self.model.jnt_qposadr[jnt_id] == qpos_idx:
                        joint_id = jnt_id
                        break

                if joint_id != -1:
                    qvel_addr = self.model.jnt_dofadr[joint_id]
                    lin_vel = self.data.qvel[qvel_addr:qvel_addr+3]
                    ang_vel = self.data.qvel[qvel_addr+3:qvel_addr+6]
                else:
                    lin_vel = np.zeros(3)
                    ang_vel = np.zeros(3)

                # Convert quaternion from [w, x, y, z] to [x, y, z, w] for messages
                quat_xyzw = (float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]))

                obj_state = ObjectState(
                    name=obj_name,
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    orientation=quat_xyzw,
                    linear_velocity=(float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])),
                    angular_velocity=(float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])),
                    timestamp=self._time
                )
                objects[obj_name] = obj_state

        # Publish scene objects message
        from ..core.bus import Topics
        scene_objects = SceneObjects(objects=objects, timestamp=self._time)
        self.bus.publish(Topics.SCENE_OBJECTS, scene_objects)

    def _start_simulation(self):
        """Start simulation loop in separate thread (runs independently at 1kHz)."""
        def simulation_loop():
            self._sim_running = True
            last_step_time = time.time()

            print("[MuJoCoSim] Simulation thread started (1kHz)")

            while self._sim_running:
                # Step simulation with lock
                with self._data_lock:
                    if self.model and self.data:
                        mujoco.mj_step(self.model, self.data)
                        self._time += self._dt
                        self._frame_id = (self._frame_id + 1) % 65536

                # Publish object states (rate-limited to 20Hz)
                self._publish_object_states()

                # Real-time sync: sleep to maintain 1kHz
                current_time = time.time()
                elapsed = current_time - last_step_time
                remaining = self._dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                last_step_time = time.time()

            print("[MuJoCoSim] Simulation thread stopped")

        self._sim_thread = threading.Thread(target=simulation_loop, daemon=True, name="MuJoCoSimulation")
        self._sim_thread.start()

    def _start_viewer(self):
        """
        Start passive viewer in separate process.

        The viewer runs in a completely separate process to avoid OpenGL
        context conflicts with the Viser GUI server. Joint positions are
        shared via shared memory for real-time visualization.
        """
        if not self.render_enabled or self._viewer_running or self.model is None:
            return

        try:
            # Create shared memory for joint positions
            nq = self.model.nq
            self._viewer_qpos_shared = mp.Array('d', nq)

            # Initialize with current position
            with self._viewer_qpos_shared.get_lock():
                qpos_np = np.frombuffer(self._viewer_qpos_shared.get_obj(), dtype=np.float64)
                qpos_np[:] = self.data.qpos[:]

            # Create stop event for inter-process communication
            self._viewer_stop_event = mp.Event()

            # Launch viewer in separate process
            self._viewer_process = mp.Process(
                target=_viewer_process_main,
                args=(self.model_path, self._viewer_stop_event, self._viewer_qpos_shared),
                daemon=True,
                name="MuJoCoViewerProcess"
            )
            self._viewer_process.start()
            self._viewer_running = True
            print("[MuJoCoSim] Viewer process started (PID: {})".format(self._viewer_process.pid))

        except Exception as e:
            print(f"[MuJoCoSim] Viewer launch error: {e}")
            self._viewer_running = False

    def start_viewer(self):
        """Public method to start viewer after GUI initialization."""
        self._start_viewer()

    def sync_viewer(self):
        """
        Sync viewer with current simulation state via shared memory.

        Updates the shared memory array with current joint positions so
        the viewer process can display the robot in its current state.
        """
        if self._viewer_running and self._viewer_qpos_shared is not None and self.data is not None:
            try:
                # Update shared memory with current joint positions
                with self._viewer_qpos_shared.get_lock():
                    qpos_np = np.frombuffer(self._viewer_qpos_shared.get_obj(), dtype=np.float64)
                    with self._data_lock:
                        qpos_np[:] = self.data.qpos[:]
            except Exception as e:
                # Silently fail - viewer might be shutting down
                pass

    def close_viewer(self):
        """Close the viewer process."""
        if self._viewer_running and self._viewer_stop_event is not None:
            print("[MuJoCoSim] Stopping viewer process...")
            self._viewer_stop_event.set()

            if self._viewer_process is not None:
                self._viewer_process.join(timeout=2.0)
                if self._viewer_process.is_alive():
                    print("[MuJoCoSim] Force terminating viewer process...")
                    self._viewer_process.terminate()
                    self._viewer_process.join(timeout=1.0)

            self._viewer_process = None
            self._viewer_stop_event = None
            self._viewer_running = False
            print("[MuJoCoSim] Viewer process stopped")

    def disconnect(self) -> None:
        """Close simulation and cleanup."""
        print("[MuJoCoSim] Disconnecting...")

        # Stop simulation thread first
        if self._sim_running:
            self._sim_running = False
            if self._sim_thread:
                self._sim_thread.join(timeout=1.0)

        # Close viewer
        self.close_viewer()

        # Clear references
        self.model = None
        self.data = None
        self._is_ready = False
        self._in_torque_mode = False

        print("[MuJoCoSim] Disconnected")

    def clear_faults(self) -> None:
        """Clear faults (no-op for simulation)."""
        pass

    def stop(self) -> None:
        """Stop motion (set zero control)."""
        if self.data is not None:
            self.data.ctrl[:7] = 0.0

    def set_servoing_mode(self, low_level: bool) -> None:
        """
        Set servoing mode (no-op for simulation, just log).

        Args:
            low_level: True for low-level mode
        """
        mode = "LOW_LEVEL" if low_level else "SINGLE_LEVEL"
        print(f"[MuJoCoSim] Servoing mode: {mode}")

    def set_torque_mode(self, enabled: bool) -> None:
        """
        Switch between torque and position control.

        Args:
            enabled: True for torque mode, False for position mode
        """
        self._in_torque_mode = enabled
        mode = "TORQUE" if enabled else "POSITION"
        print(f"[MuJoCoSim] Control mode: {mode}")

        if enabled and self.data is not None:
            # When switching TO torque mode, disable position actuators
            # but keep the robot in place by applying gravity compensation
            with self._data_lock:
                self.data.ctrl[:] = 0  # Disable position actuators
                # Compute and apply gravity compensation to prevent collapse
                mujoco.mj_inverse(self.model, self.data)
                gravity_torques = self.data.qfrc_bias[:7].copy()  # Gravity + Coriolis
                self.data.qfrc_applied[:7] = gravity_torques
                print(f"[MuJoCoSim] Applied gravity compensation: {gravity_torques}")
        elif not enabled and self.data is not None:
            # When switching to position mode, reset control to current position
            with self._data_lock:
                self.data.ctrl[:7] = self.data.qpos[:7].copy()
                self.data.qfrc_applied[:] = 0  # Clear applied torques

    def is_arm_ready(self) -> bool:
        """Check if arm is ready."""
        return self._is_ready and self.model is not None

    def wait_for_arm_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait for arm to be ready.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if ready, False if timeout
        """
        start = time.time()
        while (time.time() - start) < timeout:
            if self.is_arm_ready():
                return True
            time.sleep(0.01)
        return False

    def read_feedback(self) -> Optional[RobotFeedback]:
        """
        Read current simulation state WITHOUT stepping.

        Just reads the current state. Stepping happens in send_positions/send_torques.

        Returns:
            RobotFeedback with current joint states
        """
        if self.model is None or self.data is None:
            return None

        try:
            with self._data_lock:
                # Extract joint states (first 7 joints) - NO stepping here
                positions_deg = np.degrees(self.data.qpos[:7].copy())
                velocities_deg = np.degrees(self.data.qvel[:7].copy())

                # Measured torques from simulation
                # Use actuator forces or joint torques depending on model
                if self.model.nu >= 7:
                    torques = self.data.actuator_force[:7].copy()
                else:
                    torques = self.data.qfrc_actuator[:7].copy()

            # Get gripper position (actuator 7, which is index 7)
            gripper_pos = 0.0
            if self.model.nu >= 8 and len(self.data.qpos) > 7:
                gripper_pos = float(self.data.qpos[7])

            return RobotFeedback(
                positions_deg=positions_deg,
                velocities_deg=velocities_deg,
                torques_measured=torques,
                timestamp=self._time,
                gripper_position=gripper_pos
            )

        except Exception as e:
            print(f"[MuJoCoSim] Read error: {e}")
            return None

    def send_torques(
        self,
        torques: np.ndarray,
        positions_deg: np.ndarray
    ) -> Optional[RobotFeedback]:
        """
        Apply torques (simulation thread handles stepping).

        Args:
            torques: Joint torques (7,) in Nm (should include gravity compensation from controller)
            positions_deg: Current positions (used for tracking, not control)

        Returns:
            RobotFeedback with current state
        """
        if self.model is None or self.data is None:
            return None

        try:
            with self._data_lock:
                # In torque mode, disable position actuators and apply torques directly to joints
                self.data.ctrl[:] = 0  # Disable position actuators

                # The input torques should already include gravity compensation from ControlActor
                # Just apply them directly
                self.data.qfrc_applied[:7] = torques

            # Store positions
            self._last_positions_deg = positions_deg.copy()

            return self.read_feedback()

        except Exception as e:
            print(f"[MuJoCoSim] Send torques error: {e}")
            return None

    def send_positions(self, positions_deg: np.ndarray, gripper_pos: float = 0.0) -> Optional[RobotFeedback]:
        """
        Send position command (simulation thread handles stepping).

        This mimics real hardware: set the target, the simulation/robot runs independently.

        Args:
            positions_deg: Joint positions (7,) in degrees
            gripper_pos: Gripper position (0-255)

        Returns:
            RobotFeedback with current state
        """
        if self.model is None or self.data is None:
            return None

        try:
            with self._data_lock:
                # Set control targets to desired positions (in radians)
                positions_rad = np.radians(positions_deg)
                self.data.ctrl[:7] = positions_rad

                # Set gripper actuator (actuator 7, which is index 7)
                if self.model.nu >= 8:
                    self.data.ctrl[7] = gripper_pos

                self.data.qfrc_applied[:] = 0  # Clear any applied torques

            self._last_positions_deg = positions_deg.copy()

            return self.read_feedback()

        except Exception as e:
            print(f"[MuJoCoSim] Send positions error: {e}")
            return None

    def execute_joint_action(
        self,
        target_positions_deg: np.ndarray,
        duration: float = 8.0,
        callback=None,
    ) -> bool:
        """
        Execute smooth joint motion to target.

        Simulates high-level action by interpolating to target over duration.

        Args:
            target_positions_deg: Target joint angles (7,) in degrees
            duration: Motion duration in seconds
            callback: Optional progress callback

        Returns:
            True if successful
        """
        if self.model is None or self.data is None:
            return False

        try:
            print(f"[MuJoCoSim] Executing joint action over {duration}s")

            # Get current positions
            start_positions_deg = np.degrees(self.data.qpos[:7].copy())

            # Number of steps
            num_steps = int(duration / self._dt)

            # Interpolate and execute
            for i in range(num_steps):
                # Linear interpolation
                alpha = (i + 1) / num_steps
                current_target = (1 - alpha) * start_positions_deg + alpha * target_positions_deg

                # Send position command
                self.send_positions(current_target)

                # Optional callback
                if callback is not None and i % 100 == 0:
                    callback(alpha)

                # Sleep to match simulation rate (critical for smooth motion)
                time.sleep(self._dt)

            print(f"[MuJoCoSim] Joint action completed")
            return True

        except Exception as e:
            print(f"[MuJoCoSim] Action error: {e}")
            return False

    @property
    def in_torque_mode(self) -> bool:
        """Check if in torque mode."""
        return self._in_torque_mode
