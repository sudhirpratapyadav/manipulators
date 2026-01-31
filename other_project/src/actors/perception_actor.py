"""
Perception Actor - base class for camera and sensor actors.

Provides framework for integrating cameras, depth sensors, and other perception systems.
Subclasses implement specific camera interfaces (RealSense, webcam, MuJoCo camera, etc.)
"""

import queue
import time
from abc import abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from ..core.actor import TimedActor
from ..core.bus import MessageBus, Topics
from ..core.messages import ImageMessage, DepthMessage, PointCloudMessage
from ..core.config import Config


class PerceptionActor(TimedActor):
    """
    Base class for perception actors (cameras, sensors).

    Publishes:
        - /perception/image/{camera_id}
        - /perception/depth/{camera_id}
        - /perception/pointcloud/{camera_id}
    """

    def __init__(self, camera_id: str, bus: MessageBus, config: Config, rate_hz: int = 30):
        super().__init__(
            name=f"PerceptionActor_{camera_id}",
            bus=bus,
            config=config,
            rate_hz=rate_hz
        )

        self.camera_id = camera_id
        self._config = config

        # Topics for this camera
        self.image_topic = Topics.PERCEPTION_IMAGE.replace("{camera_id}", camera_id)
        self.depth_topic = Topics.PERCEPTION_DEPTH.replace("{camera_id}", camera_id)
        self.pointcloud_topic = Topics.PERCEPTION_POINTCLOUD.replace("{camera_id}", camera_id)

    @abstractmethod
    def setup_camera(self) -> bool:
        """
        Initialize camera hardware/connection.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def capture_frame(self) -> Dict[str, Any]:
        """
        Capture a single frame from the camera.

        Returns:
            Dictionary with keys:
                - 'rgb': numpy array (H, W, 3) if RGB available
                - 'depth': numpy array (H, W) if depth available
                - 'success': bool indicating if capture succeeded
        """
        pass

    @abstractmethod
    def cleanup_camera(self) -> None:
        """Cleanup camera resources."""
        pass

    def setup(self) -> None:
        """Initialize perception actor."""
        success = self.setup_camera()
        if not success:
            print(f"[{self.name}] Failed to setup camera")

    def loop(self) -> None:
        """Capture and publish perception data."""
        frame = self.capture_frame()

        if not frame.get('success', False):
            return

        # Publish RGB image if available
        if 'rgb' in frame and frame['rgb'] is not None:
            rgb = frame['rgb']
            msg = ImageMessage(
                camera_id=self.camera_id,
                width=rgb.shape[1],
                height=rgb.shape[0],
                channels=rgb.shape[2] if len(rgb.shape) == 3 else 1,
                encoding="rgb8",
                data=rgb.tobytes()
            )
            self.bus.publish(self.image_topic, msg)

        # Publish depth if available
        if 'depth' in frame and frame['depth'] is not None:
            depth = frame['depth']
            msg = DepthMessage(
                camera_id=self.camera_id,
                width=depth.shape[1],
                height=depth.shape[0],
                encoding="32FC1",
                data=depth.tobytes()
            )
            self.bus.publish(self.depth_topic, msg)

        # Publish point cloud if available
        if 'pointcloud' in frame and frame['pointcloud'] is not None:
            pc = frame['pointcloud']  # Expected: (N, 3) or (N, 6) with colors
            points = tuple(tuple(p[:3]) for p in pc)
            colors = tuple(tuple(p[3:6].astype(int)) for p in pc) if pc.shape[1] >= 6 else ()

            msg = PointCloudMessage(
                camera_id=self.camera_id,
                num_points=len(points),
                points=points,
                colors=colors
            )
            self.bus.publish(self.pointcloud_topic, msg)

    def teardown(self) -> None:
        """Cleanup perception resources."""
        self.cleanup_camera()


# ============ Example Implementations ============

class MockCameraActor(PerceptionActor):
    """Mock camera for testing."""

    def __init__(self, camera_id: str, bus: MessageBus, config: Config):
        super().__init__(camera_id, bus, config, rate_hz=10)

    def setup_camera(self) -> bool:
        print(f"[MockCamera-{self.camera_id}] Mock camera initialized")
        return True

    def capture_frame(self) -> Dict[str, Any]:
        # Generate test pattern
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return {'rgb': rgb, 'success': True}

    def cleanup_camera(self) -> None:
        print(f"[MockCamera-{self.camera_id}] Cleanup")


class MuJoCoOffboardCameraActor(PerceptionActor):
    """
    Offboard camera actor for MuJoCo simulation.

    Renders camera views from the simulation for perception tasks.
    """

    def __init__(self, camera_id: str, bus: MessageBus, config: Config,
                 mujoco_sim=None, camera_name: str = "offboard_camera"):
        super().__init__(camera_id, bus, config, rate_hz=30)
        self.mujoco_sim = mujoco_sim  # Reference to MuJoCoSimulator
        self.camera_name = camera_name
        self.renderer = None

    def setup_camera(self) -> bool:
        """Setup MuJoCo camera renderer."""
        if self.mujoco_sim is None:
            print(f"[MuJoCoCamera-{self.camera_id}] No simulation instance provided")
            return False

        try:
            import mujoco
            # Setup renderer for this camera
            # (Implementation depends on MuJoCo version and rendering backend)
            print(f"[MuJoCoCamera-{self.camera_id}] MuJoCo camera '{self.camera_name}' initialized")
            return True
        except Exception as e:
            print(f"[MuJoCoCamera-{self.camera_id}] Setup error: {e}")
            return False

    def capture_frame(self) -> Dict[str, Any]:
        """Render frame from MuJoCo camera."""
        if self.mujoco_sim is None:
            return {'success': False}

        try:
            # Render RGB and depth from simulation
            # (Implementation depends on MuJoCo camera API)
            # This is a placeholder - actual implementation would call mujoco.mjr_render
            rgb = None  # Would be: renderer.render(width, height, camera_id)
            depth = None  # Would be: renderer.render_depth(...)

            return {
                'rgb': rgb,
                'depth': depth,
                'success': rgb is not None
            }
        except Exception as e:
            print(f"[MuJoCoCamera-{self.camera_id}] Capture error: {e}")
            return {'success': False}

    def cleanup_camera(self) -> None:
        """Cleanup renderer."""
        if self.renderer is not None:
            # Cleanup renderer resources
            pass
        print(f"[MuJoCoCamera-{self.camera_id}] Cleanup")


class RealSenseCameraActor(PerceptionActor):
    """
    Intel RealSense camera actor.

    Requires pyrealsense2 library.
    """

    def __init__(self, camera_id: str, bus: MessageBus, config: Config):
        super().__init__(camera_id, bus, config, rate_hz=30)
        self.pipeline = None
        self.align = None

    def setup_camera(self) -> bool:
        """Initialize RealSense camera."""
        try:
            import pyrealsense2 as rs

            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

            print(f"[RealSense-{self.camera_id}] Camera initialized")
            return True
        except Exception as e:
            print(f"[RealSense-{self.camera_id}] Setup error: {e}")
            return False

    def capture_frame(self) -> Dict[str, Any]:
        """Capture RGB-D frame."""
        try:
            import pyrealsense2 as rs
            import numpy as np

            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return {'success': False}

            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # mm to meters

            return {'rgb': rgb, 'depth': depth, 'success': True}
        except Exception as e:
            print(f"[RealSense-{self.camera_id}] Capture error: {e}")
            return {'success': False}

    def cleanup_camera(self) -> None:
        """Stop pipeline."""
        if self.pipeline is not None:
            self.pipeline.stop()
        print(f"[RealSense-{self.camera_id}] Cleanup")
