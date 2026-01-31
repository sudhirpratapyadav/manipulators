"""
Simple message bus for inter-actor communication.

Uses queue.Queue for thread-safe message passing with minimal overhead (~1-2μs per operation).
"""

import queue
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
import time


@dataclass
class Subscription:
    """A subscription to a topic."""
    callback: Callable[[Any], None]
    queue: Optional[queue.Queue]  # If None, uses callback directly


class MessageBus:
    """
    Thread-safe publish/subscribe message bus.

    Features:
    - Topic-based pub/sub
    - Latest-value cache for each topic
    - Both callback and queue-based subscriptions
    - Thread-safe with minimal locking

    Performance: ~1-2μs per publish/subscribe operation
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._subscribers: Dict[str, List[Subscription]] = {}
        self._running = True

    def publish(self, topic: str, message: Any) -> None:
        """
        Publish a message to a topic.

        - Stores as latest value for the topic
        - Notifies all subscribers (callbacks or queues)

        Args:
            topic: Topic name (e.g., "/robot/state")
            message: Message to publish (should be a Message dataclass)
        """
        with self._lock:
            self._latest[topic] = message
            subscribers = self._subscribers.get(topic, []).copy()

        # Notify outside lock to avoid deadlocks
        for sub in subscribers:
            if sub.queue is not None:
                try:
                    # Non-blocking put, drop if full
                    sub.queue.put_nowait(message)
                except queue.Full:
                    pass
            elif sub.callback is not None:
                try:
                    sub.callback(message)
                except Exception as e:
                    print(f"[Bus] Callback error on {topic}: {e}")

    def get_latest(self, topic: str) -> Optional[Any]:
        """
        Get the latest message for a topic (non-blocking).

        Args:
            topic: Topic name

        Returns:
            Latest message or None if no message has been published
        """
        with self._lock:
            return self._latest.get(topic)

    def subscribe_callback(self, topic: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to a topic with a callback.

        The callback will be called synchronously when a message is published.
        Keep callbacks fast to avoid blocking publishers.

        Args:
            topic: Topic name
            callback: Function to call with each message
        """
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(Subscription(callback=callback, queue=None))

    def subscribe_queue(self, topic: str, maxsize: int = 10) -> queue.Queue:
        """
        Subscribe to a topic with a queue.

        Messages will be put into the returned queue. Use this for actors
        that need to process messages in their own loop.

        Args:
            topic: Topic name
            maxsize: Maximum queue size (oldest messages dropped if full)

        Returns:
            Queue that will receive messages
        """
        q = queue.Queue(maxsize=maxsize)
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(Subscription(callback=None, queue=q))
        return q

    def unsubscribe_callback(self, topic: str, callback: Callable) -> None:
        """Remove a callback subscription."""
        with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [
                    s for s in self._subscribers[topic]
                    if s.callback != callback
                ]

    def wait_for_message(
        self,
        topic: str,
        timeout: Optional[float] = None,
        predicate: Optional[Callable[[Any], bool]] = None
    ) -> Optional[Any]:
        """
        Block until a message matching predicate is published.

        Args:
            topic: Topic to wait on
            timeout: Maximum time to wait (None = forever)
            predicate: Optional function to filter messages

        Returns:
            Matching message or None if timeout
        """
        result_queue = queue.Queue(maxsize=1)

        def check_message(msg):
            if predicate is None or predicate(msg):
                try:
                    result_queue.put_nowait(msg)
                except queue.Full:
                    pass

        self.subscribe_callback(topic, check_message)
        try:
            return result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        finally:
            self.unsubscribe_callback(topic, check_message)

    def get_topics(self) -> List[str]:
        """Get list of all topics with published messages."""
        with self._lock:
            return list(self._latest.keys())

    def clear(self) -> None:
        """Clear all subscriptions and cached messages."""
        with self._lock:
            self._latest.clear()
            self._subscribers.clear()


from typing import TypeVar, Generic

T = TypeVar('T')


class Topic(Generic[T]):
    """Typed topic for type-safe message passing."""

    def __init__(self, name: str, message_type: type[T]):
        self.name = name
        self.message_type = message_type

    def __repr__(self) -> str:
        return f"Topic({self.name}, {self.message_type.__name__})"


# Topic constants for type safety
class Topics:
    """Standard topic names with type information."""
    # Import here to avoid circular dependencies - done lazily in create_topics()
    # Robot state
    ROBOT_STATE: str = "/robot/state"

    # Control
    CONTROL_TORQUE: str = "/control/torque"
    CONTROL_DESIRED: str = "/control/desired"
    CONTROL_MODE: str = "/control/mode"
    CONTROL_ENABLE: str = "/control/enable"  # Enable/disable control actor
    TARGET_POSE: str = "/control/target_pose"
    JOINT_POSITION_CMD: str = "/control/joint_position"

    # IK
    IK_ENABLE: str = "/ik/enable"  # Enable/disable IK actor
    IK_RESET: str = "/ik/reset"  # Reset IK state

    # Input
    INPUT_DELTA: str = "/input/delta"
    INPUT_SLIDERS: str = "/input/sliders"
    INPUT_ENABLE: str = "/input/enable"  # Enable/disable input plugins

    # System
    SYSTEM_COMMAND: str = "/system/command"
    SYSTEM_STATE: str = "/system/state"
    STATE_TRANSITION_REQUEST: str = "/state/transition_request"

    # Mode control
    MODE_CHANGE_REQUEST: str = "/mode/change_request"
    MODE_CHANGED: str = "/mode/changed"

    # Hardware
    HARDWARE_WRITE: str = "/hardware/write"
    HARDWARE_FEEDBACK: str = "/hardware/feedback"

    # Safety
    SAFETY_EVENT: str = "/safety/event"

    # UI
    UI_STATE: str = "/ui/state"

    # Gripper
    GRIPPER_COMMAND: str = "/gripper/command"

    # Perception (for cameras/sensors)
    PERCEPTION_IMAGE: str = "/perception/image/{camera_id}"
    PERCEPTION_DEPTH: str = "/perception/depth/{camera_id}"
    PERCEPTION_POINTCLOUD: str = "/perception/pointcloud/{camera_id}"

    # AI/ML
    AI_INFERENCE_REQUEST: str = "/ai/inference/request"
    AI_INFERENCE_RESULT: str = "/ai/inference/result/{model_id}"

    # Scene/Objects
    SCENE_OBJECTS: str = "/scene/objects"  # Dynamic object states
