"""
RealSense Perception Actor with Object Detection.

Streams RGB-D data from Intel RealSense camera and publishes detected object poses.
Can run in a separate process for better performance.
"""

from typing import Dict, Tuple
import numpy as np
import multiprocessing as mp

from ..core.actor import TimedActor
from ..core.bus import MessageBus, Topics
from ..core.messages import ImageMessage, DepthMessage, SceneObjects, ObjectState
from ..core.config import Config


def dummy_object_detector(rgb: np.ndarray, depth: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:  # noqa: ARG001
    """
    Dummy object detector for testing.

    In a real implementation, this would use a deep learning model to detect objects
    and estimate their 6D poses from RGB-D data.

    Args:
        rgb: RGB image (H, W, 3) uint8
        depth: Depth image (H, W) float32 in meters

    Returns:
        Dictionary mapping object names to (position, orientation) tuples
        - position: (x, y, z) in meters in camera frame
        - orientation: quaternion (x, y, z, w)
    """
    # Dummy implementation: Returns a fixed pose for a single object
    # In reality, this would run object detection + 6D pose estimation

    # Example: Assume we detect a cube at a fixed location in camera frame
    # Camera frame: X right, Y down, Z forward
    # Object is 0.5m in front of camera, slightly to the right and above center

    detected_objects = {
        "cube1": (
            np.array([0.1, -0.05, 0.5]),  # position: 10cm right, 5cm up, 50cm forward
            np.array([0.0, 0.0, 0.0, 1.0])  # orientation: identity quaternion
        )
    }

    return detected_objects


class RealSensePerceptionActor(TimedActor):
    """
    RealSense camera actor with object detection.

    Publishes:
        - /perception/image/realsense: RGB images
        - /perception/depth/realsense: Depth images
        - /scene/objects: Detected object poses
    """

    def __init__(self, bus: MessageBus, config: Config, camera_id: str = "realsense"):
        super().__init__(
            name=f"RealSensePerception_{camera_id}",
            bus=bus,
            config=config,
            rate_hz=30  # 30 Hz camera capture
        )

        self.camera_id = camera_id
        self._config = config

        # RealSense pipeline
        self.pipeline = None
        self.align = None

        # Camera intrinsics (will be set during setup)
        self.intrinsics = None

        # Topics
        self.image_topic = Topics.PERCEPTION_IMAGE.replace("{camera_id}", camera_id)
        self.depth_topic = Topics.PERCEPTION_DEPTH.replace("{camera_id}", camera_id)
        self.objects_topic = Topics.SCENE_OBJECTS

    def setup(self) -> None:
        """Initialize RealSense camera."""
        try:
            import pyrealsense2 as rs

            print(f"[{self.name}] Initializing RealSense camera...")

            self.pipeline = rs.pipeline()
            config = rs.config()

            # Configure streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            # Start pipeline
            profile = self.pipeline.start(config)

            # Create align object to align depth to color
            self.align = rs.align(rs.stream.color)

            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

            print(f"[{self.name}] RealSense camera initialized successfully")
            print(f"[{self.name}] Resolution: {self.intrinsics.width}x{self.intrinsics.height}")
            print(f"[{self.name}] Focal length: ({self.intrinsics.fx:.1f}, {self.intrinsics.fy:.1f})")

        except ImportError:
            print(f"[{self.name}] ERROR: pyrealsense2 not installed. Install with: pip install pyrealsense2")
            raise
        except Exception as e:
            print(f"[{self.name}] ERROR: Failed to initialize RealSense camera: {e}")
            raise

    def loop(self) -> None:
        """Capture frame, detect objects, and publish results."""
        try:
            import pyrealsense2 as rs

            # Wait for frames
            frames = self.pipeline.wait_for_frames()

            # Align depth to color
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return

            # Convert to numpy arrays
            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # mm to meters

            # Publish RGB image
            rgb_msg = ImageMessage(
                camera_id=self.camera_id,
                width=rgb.shape[1],
                height=rgb.shape[0],
                channels=rgb.shape[2],
                encoding="rgb8",
                data=rgb.tobytes()
            )
            self.bus.publish(self.image_topic, rgb_msg)

            # Publish depth image
            depth_msg = DepthMessage(
                camera_id=self.camera_id,
                width=depth.shape[1],
                height=depth.shape[0],
                encoding="32FC1",
                data=depth.tobytes()
            )
            self.bus.publish(self.depth_topic, depth_msg)

            # Run object detection
            detected_objects = dummy_object_detector(rgb, depth)

            # Convert detected objects to camera frame poses and publish
            if detected_objects:
                # Transform from camera frame to world frame
                # TODO: This requires camera calibration/mounting information
                # For now, we publish in camera frame

                object_states = {}
                for obj_name, (position, orientation) in detected_objects.items():
                    obj_state = ObjectState(
                        name=obj_name,
                        position=tuple(position),
                        orientation=tuple(orientation),
                        linear_velocity=(0.0, 0.0, 0.0),
                        angular_velocity=(0.0, 0.0, 0.0)
                    )
                    object_states[obj_name] = obj_state

                # Publish scene objects
                scene_msg = SceneObjects(objects=object_states)
                self.bus.publish(self.objects_topic, scene_msg)

        except Exception as e:
            print(f"[{self.name}] Error in loop: {e}")

    def teardown(self) -> None:
        """Stop pipeline and cleanup."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                print(f"[{self.name}] Pipeline stopped")
            except Exception as e:
                print(f"[{self.name}] Error stopping pipeline: {e}")


def run_perception_actor_process(image_queue: mp.Queue, objects_queue: mp.Queue, config: Config, camera_id: str, stop_event: mp.Event) -> None:
    """
    Run perception actor in a separate process.

    Args:
        image_queue: Queue for sending image messages to main process
        objects_queue: Queue for sending object detections to main process
        config: Configuration
        camera_id: Camera identifier
        stop_event: Event to signal process termination
    """
    print(f"[PerceptionProcess] Starting perception actor in separate process (PID: {mp.current_process().pid})")

    try:
        import pyrealsense2 as rs

        # Setup RealSense pipeline
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = pipeline.start(rs_config)
        align = rs.align(rs.stream.color)

        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        print(f"[PerceptionProcess] RealSense camera initialized successfully")
        print(f"[PerceptionProcess] Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"[PerceptionProcess] Focal length: ({intrinsics.fx:.1f}, {intrinsics.fy:.1f})")

        # Main loop
        while not stop_event.is_set():
            try:
                # Wait for frames with timeout
                frames = pipeline.wait_for_frames(timeout_ms=100)

                # Align depth to color
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert to numpy arrays
                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # mm to meters

                # Create and send image message (non-blocking)
                from ..core.messages import ImageMessage, DepthMessage, SceneObjects, ObjectState

                rgb_msg = {
                    'camera_id': camera_id,
                    'width': rgb.shape[1],
                    'height': rgb.shape[0],
                    'channels': rgb.shape[2],
                    'encoding': 'rgb8',
                    'data': rgb.tobytes()
                }

                try:
                    image_queue.put_nowait(rgb_msg)
                except:
                    pass  # Queue full, skip this frame

                # Run object detection
                detected_objects = dummy_object_detector(rgb, depth)

                if detected_objects:
                    # Send object detections
                    obj_dict = {}
                    for obj_name, (position, orientation) in detected_objects.items():
                        obj_dict[obj_name] = {
                            'name': obj_name,
                            'position': tuple(position),
                            'orientation': tuple(orientation),
                            'linear_velocity': (0.0, 0.0, 0.0),
                            'angular_velocity': (0.0, 0.0, 0.0)
                        }

                    try:
                        objects_queue.put_nowait({'objects': obj_dict})
                    except:
                        pass  # Queue full, skip

            except Exception as e:
                if "Timeout" not in str(e):
                    print(f"[PerceptionProcess] Frame capture error: {e}")

        print(f"[PerceptionProcess] Stop signal received, shutting down...")

        # Cleanup
        pipeline.stop()
        print(f"[PerceptionProcess] Pipeline stopped")

    except Exception as e:
        print(f"[PerceptionProcess] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[PerceptionProcess] Process terminated")


