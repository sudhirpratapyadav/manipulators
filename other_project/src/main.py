"""
Main orchestrator for the Kinova Gen3 controller.

Coordinates all actors and handles state transitions.
"""

import time
import argparse
from threading import Thread
from typing import Optional
from pathlib import Path
import numpy as np
import multiprocessing as mp

from .core.bus import MessageBus, Topics
from .core.config import Config, load_config
from .core.messages import ControllerState, ControlMode, ControlModeChange, StateChange
from .core.scene_config import load_scene_config
from .core.scene_builder import SceneBuilder

from .actors.hardware_actor import HardwareActor
from .actors.control_actor import ControlActor
from .actors.ik_actor import IKActor
from .actors.safety_actor import SafetyActor
from .actors.state_actor import StateActor
from .actors.realsense_perception_actor import RealSensePerceptionActor, run_perception_actor_process

from .inputs.keyboard import KeyboardInput

from .gui.server import GUIServer
# RobotModel imported lazily to avoid pinocchio dependency


class KinovaController:
    """
    Main controller orchestrating all actors and inputs.

    Responsibilities:
    - Create and manage all actors
    - Handle high-level state transitions
    - Coordinate mode changes
    - Run main update loop
    """

    def __init__(self, config: Config):
        self.config = config

        # Core components
        self.bus = MessageBus()

        # Lazy load RobotModel to avoid pinocchio dependency
        from .robot.model import RobotModel
        self.model = RobotModel()

        # Scene config (loaded later in setup)
        self.scene = None

        # Actors
        self.hardware = HardwareActor(self.bus, config)
        self.control = ControlActor(self.bus, config)
        self.ik = IKActor(self.bus, config)
        self.safety = SafetyActor(self.bus, config)
        self.state = StateActor(self.bus, config)

        # Perception actor (created on demand via GUI)
        self.perception = None
        self.perception_process = None
        self.perception_stop_event = None
        self.perception_image_queue = None
        self.perception_objects_queue = None

        # Input plugins
        self.keyboard = KeyboardInput(self.bus, {
            "position_step_m": config.inputs.keyboard.position_step_m,
            "rotation_step_rad": config.inputs.keyboard.rotation_step_rad,
        })

        # GUI
        self.gui = GUIServer(self.bus, config)

        # State
        self._running = False
        self._current_state = ControllerState.DISCONNECTED

    def setup(self) -> None:
        """Initialize all components."""
        print("[Controller] Setting up...")

        # Setup GUI
        self.gui.setup()
        self.gui.set_callback("on_connect", self.on_connect)
        self.gui.set_callback("on_disconnect", self.on_disconnect)
        self.gui.set_callback("on_go_home", self.on_go_home)
        self.gui.set_callback("on_joint_control_start", self.on_joint_control_start)
        self.gui.set_callback("on_diffik_start", self.on_diffik_start)
        self.gui.set_callback("on_stop", self.on_stop)
        self.gui.set_callback("on_camera_toggle", self.on_camera_toggle)

        # Load scene if using simulation (must be before hardware.connect())
        if self.config.simulation.enabled:
            self._load_scene()
            # Pass scene config to hardware actor
            if self.scene is not None:
                self.hardware.set_scene_config(self.scene)

        # Start state actor (needed for UI)
        self.state.start()

        # Start input plugins (they filter based on enabled state)
        self.keyboard.start()

        self._set_state(ControllerState.DISCONNECTED, "Ready to connect")
        print("[Controller] Setup complete")

    def run(self) -> None:
        """Main loop at 50Hz."""
        self._running = True
        period = 1.0 / self.config.control.rates.gui_hz

        print("[Controller] Running main loop...")
        try:
            while self._running:
                t_start = time.time()

                # Forward messages from perception process to bus
                self._forward_perception_messages()

                # Update GUI
                self.gui.update()

                # Sync MuJoCo viewer (if simulation is running)
                self.hardware.sync_viewer()

                # Maintain rate
                elapsed = time.time() - t_start
                if elapsed < period:
                    time.sleep(period - elapsed)

        except KeyboardInterrupt:
            print("\n[Controller] Interrupted")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Cleanup all components."""
        print("[Controller] Shutting down...")
        self._running = False

        # Stop control first
        self.on_stop()

        # Stop actors
        self.safety.stop()
        self.ik.stop()
        self.control.stop()
        self.hardware.stop()
        self.state.stop()

        # Stop perception process
        if self.perception_process is not None and self.perception_process.is_alive():
            print("[Controller] Stopping perception process...")
            if self.perception_stop_event is not None:
                self.perception_stop_event.set()
            self.perception_process.join(timeout=2.0)
            if self.perception_process.is_alive():
                self.perception_process.terminate()

        # Stop inputs
        self.keyboard.stop()

        # Disconnect hardware
        self.hardware.disconnect()

        print("[Controller] Shutdown complete")

    def _set_state(self, state: ControllerState, message: str = "") -> None:
        """Update controller state."""
        self._current_state = state
        self.state.set_state(state, message)

    def _load_scene(self) -> None:
        """Load scene configuration and update simulation/visualization."""
        try:
            # Load scene config from config file
            scene_config_path = Path(self.config.simulation.scene_config)
            if not scene_config_path.exists():
                print(f"[Controller] Scene config not found at {scene_config_path}, using default empty scene")
                return

            print(f"[Controller] Loading scene from {scene_config_path}")
            self.scene = load_scene_config(str(scene_config_path))

            # Build MuJoCo scene
            assets_dir = Path("src/assets")
            builder = SceneBuilder(assets_dir)

            # Generate scene XML
            scene_output = "src/assets/robots/kinova/mjcf/scene_generated.xml"
            scene_file = builder.build_mujoco_scene(self.scene, scene_output)
            print(f"[Controller] Generated MuJoCo scene: {scene_file}")

            # Update simulation config to use generated scene
            self.config.simulation.model_path = scene_file

            # Set ground height in Viser
            ground_height = builder.get_ground_height(self.scene)
            self.gui.set_ground_height(ground_height)

            # Load all scene objects (both static and dynamic) in Viser
            scene_objects = builder.get_scene_objects(self.scene)

            # Separate static and dynamic objects
            static_objects = [obj for obj in scene_objects if obj["static"]]
            dynamic_objects = [obj for obj in scene_objects if not obj["static"]]

            # Load static objects as obstacles
            self.gui.load_scene_obstacles(static_objects)

            # Load dynamic objects (will get pose updates from sim)
            self.gui.load_dynamic_objects(dynamic_objects)

            print(f"[Controller] Scene loaded: {len(static_objects)} static, {len(dynamic_objects)} dynamic")

        except Exception as e:
            print(f"[Controller] Error loading scene: {e}")
            import traceback
            traceback.print_exc()

    # ============ CALLBACKS ============

    def on_connect(self) -> None:
        """Handle connect button."""
        if self._current_state != ControllerState.DISCONNECTED:
            return

        def connect_thread():
            self._set_state(ControllerState.CONNECTING, "Connecting...")

            if self.hardware.connect():
                # Start hardware actor loop (1kHz UDP)
                self.hardware.start()

                # Start control actors
                self.control.start()
                self.safety.start()
                self.ik.start()

                # Compute and show bounds
                home_pose = self.ik.home_pose
                if home_pose is not None:
                    bound = self.config.control.limits.position_bound_m
                    self.gui.set_position_bounds(
                        home_pose[:3] - bound,
                        home_pose[:3] + bound
                    )

                # Start viewer AFTER GUI initialization to avoid threading conflicts
                self.hardware.start_viewer()

                self._set_state(ControllerState.IDLE, "Connected - Ready")
            else:
                self._set_state(ControllerState.DISCONNECTED, "Connection failed")

        Thread(target=connect_thread, name="Connect", daemon=True).start()

    def on_disconnect(self) -> None:
        """Handle disconnect button."""
        self.on_stop()

        # Stop actors
        self.safety.stop()
        self.ik.stop()
        self.control.stop()
        self.hardware.stop()

        # Disconnect hardware
        self.hardware.disconnect()

        self._set_state(ControllerState.DISCONNECTED, "Disconnected")

    def on_go_home(self) -> None:
        """Handle go home button."""
        if self._current_state != ControllerState.IDLE:
            return

        def home_thread():
            self._set_state(ControllerState.GOING_HOME, "Moving to home...")

            # Get home position
            home_q = np.array(self.config.home_joints_rad)
            home_deg = np.degrees(home_q)

            # Execute high-level move
            success = self.hardware.execute_high_level_action(home_deg, duration=8.0)

            if success:
                self._set_state(ControllerState.IDLE, "Home position reached")
            else:
                self._set_state(ControllerState.IDLE, "Home motion failed")

        Thread(target=home_thread, name="GoHome", daemon=True).start()

    def on_joint_control_start(self) -> None:
        """Handle start joint control button."""
        if self._current_state != ControllerState.IDLE:
            return

        self._set_state(ControllerState.JOINT_CONTROL, "Joint control active - use sliders")
        # Stay in position mode - sliders will send position commands directly
        # Clear any previous slider commands
        self.hardware._last_slider_command = None

    def on_diffik_start(self) -> None:
        """Handle start diff-IK button."""
        if self._current_state != ControllerState.IDLE:
            return

        def diffik_thread():
            self._set_state(ControllerState.DIFFIK_INIT, "Initializing Diff-IK...")

            # Go to home first
            home_q = np.array(self.config.home_joints_rad)
            home_deg = np.degrees(home_q)

            success = self.hardware.execute_high_level_action(home_deg, duration=8.0)
            if not success:
                self._set_state(ControllerState.IDLE, "Failed to reach home")
                return

            # Wait a bit for position to settle
            time.sleep(0.5)

            # Get actual current position (important: robot might not be exactly at commanded home)
            latest_state = self.bus.get_latest(Topics.ROBOT_STATE)
            if latest_state is not None:
                actual_q = np.array(latest_state.joint_positions)
                print(f"[DiffIK] Actual position after going home: {np.degrees(actual_q)}")
            else:
                actual_q = home_q
                print(f"[DiffIK] Warning: No robot state available, using commanded home")

            # Enable IK and control with ACTUAL position
            self.ik.enable(initial_q=actual_q)
            self.control.enable()

            # Enable inputs
            self.keyboard.enable()

            # Enable torque mode
            self.bus.publish(Topics.CONTROL_MODE, ControlModeChange(
                mode=ControlMode.TORQUE,
                reason="Diff-IK started"
            ))

            self._set_state(ControllerState.DIFFIK_ACTIVE, "Diff-IK active - use keyboard")

        Thread(target=diffik_thread, name="DiffIKStart", daemon=True).start()

    def on_stop(self) -> None:
        """Handle stop button - stop all control."""
        print("[Controller] Stop requested")

        # Disable inputs
        self.keyboard.disable()

        # Disable IK and control
        self.ik.disable()
        self.control.disable()

        # Clear slider commands when stopping joint control
        if hasattr(self.hardware, '_last_slider_command'):
            self.hardware._last_slider_command = None

        # Switch to position mode
        self.bus.publish(Topics.CONTROL_MODE, ControlModeChange(
            mode=ControlMode.POSITION,
            reason="Stop requested"
        ))

        # Update state
        if self._current_state not in (ControllerState.DISCONNECTED, ControllerState.CONNECTING):
            self._set_state(ControllerState.IDLE, "Stopped")

    def _forward_perception_messages(self) -> None:
        """Forward messages from perception process queues to main bus."""
        # Forward image messages
        if self.perception_image_queue is not None:
            try:
                while True:
                    msg_dict = self.perception_image_queue.get_nowait()
                    from .core.messages import ImageMessage
                    msg = ImageMessage(**msg_dict)
                    self.bus.publish(Topics.PERCEPTION_IMAGE.replace("{camera_id}", self.config.perception.camera_id), msg)
            except:
                pass

        # Forward object detection messages
        if self.perception_objects_queue is not None:
            try:
                while True:
                    msg_dict = self.perception_objects_queue.get_nowait()
                    from .core.messages import ObjectState, SceneObjects

                    objects = {}
                    for name, obj_data in msg_dict['objects'].items():
                        objects[name] = ObjectState(**obj_data)

                    scene_msg = SceneObjects(objects=objects)
                    self.bus.publish(Topics.SCENE_OBJECTS, scene_msg)
            except:
                pass

    def on_camera_toggle(self) -> None:
        """Handle camera start/stop button."""
        if self.perception_process is None or not self.perception_process.is_alive():
            # Start camera in separate process
            print("[Controller] Starting RealSense camera in separate process...")

            try:
                # Create queues for inter-process communication
                self.perception_image_queue = mp.Queue(maxsize=2)
                self.perception_objects_queue = mp.Queue(maxsize=2)
                self.perception_stop_event = mp.Event()

                # Create and start the process
                self.perception_process = mp.Process(
                    target=run_perception_actor_process,
                    args=(self.perception_image_queue, self.perception_objects_queue,
                          self.config, self.config.perception.camera_id, self.perception_stop_event),
                    name="PerceptionProcess",
                    daemon=True
                )
                self.perception_process.start()

                print(f"[Controller] Camera process started (PID: {self.perception_process.pid})")
                # Notify GUI that camera is running
                self.gui.set_camera_running(True)

            except Exception as e:
                print(f"[Controller] Failed to start camera process: {e}")
                import traceback
                traceback.print_exc()
                self.perception_process = None
                self.perception_stop_event = None
                self.perception_image_queue = None
                self.perception_objects_queue = None
                # Notify GUI that camera failed to start
                self.gui.set_camera_running(False)
        else:
            # Stop camera process
            print("[Controller] Stopping camera process...")

            if self.perception_stop_event is not None:
                self.perception_stop_event.set()

            if self.perception_process is not None:
                self.perception_process.join(timeout=2.0)
                if self.perception_process.is_alive():
                    print("[Controller] Force terminating camera process...")
                    self.perception_process.terminate()
                    self.perception_process.join(timeout=1.0)

            self.perception_process = None
            self.perception_stop_event = None
            self.perception_image_queue = None
            self.perception_objects_queue = None

            print("[Controller] Camera process stopped")
            # Notify GUI that camera is stopped
            self.gui.set_camera_running(False)


def main():
    parser = argparse.ArgumentParser(description="Kinova Gen3 Controller")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--ip", type=str, help="Robot IP address")
    parser.add_argument("--sim", action="store_true", help="Use MuJoCo simulation instead of real hardware")
    parser.add_argument("--no-render", action="store_true", help="Disable simulation visualization")
    parser.add_argument("--scene", type=str, help="Path to scene config file (default: src/config/scene.yaml)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override IP if provided
    if args.ip:
        config.kinova.ip = args.ip

    # Override simulation settings if provided
    if args.sim:
        config.simulation.enabled = True
    if args.no_render:
        config.simulation.render = False
    if args.scene:
        config.simulation.scene_config = args.scene

    # Create and run controller
    controller = KinovaController(config)
    controller.setup()
    controller.run()


if __name__ == "__main__":
    main()
