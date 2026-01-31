"""
Typed message definitions for the message bus.

All messages are immutable dataclasses. Components communicate by publishing
and subscribing to these message types on specific topics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, Any
import time


class ControlMode(Enum):
    """Control mode for the robot."""
    POSITION = "position"
    TORQUE = "torque"


class ControllerState(Enum):
    """High-level controller state machine."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    IDLE = "idle"
    GOING_HOME = "going_home"
    JOINT_CONTROL = "joint_control"
    DIFFIK_INIT = "diffik_init"
    DIFFIK_ACTIVE = "diffik_active"
    ERROR = "error"


class SystemCommandType(Enum):
    """Types of system commands."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    GO_HOME = "go_home"
    START_JOINT_CONTROL = "start_joint_control"
    START_DIFFIK = "start_diffik"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True, slots=True)
class RobotState:
    """
    Robot state from hardware.
    Published by: HardwareActor
    Topic: /robot/state
    """
    joint_positions: Tuple[float, ...]  # 7 values, radians
    joint_velocities: Tuple[float, ...]  # 7 values, rad/s
    connection_healthy: bool = True
    gripper_position: float = 0.0  # Gripper position (0-255 for MuJoCo, 0-1 for real hardware)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class TorqueCommand:
    """
    Torque command to send to hardware.
    Published by: ControlActor
    Topic: /control/torque
    """
    torques: Tuple[float, ...]  # 7 values, Nm
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class DesiredJoints:
    """
    Desired joint positions for PD control.
    Published by: IKActor or SliderInput
    Topic: /control/desired
    """
    positions: Tuple[float, ...]  # 7 values, radians
    source: str = "unknown"  # "ik", "sliders", "home"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class GripperCommand:
    """
    Gripper position command.
    Published by: GUI
    Topic: /gripper/command
    """
    position: float  # Gripper position (0-255 for MuJoCo, 0-1 normalized for real hardware)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PoseDelta:
    """
    Pose delta from any input source.
    Published by: InputActor (from keyboard, OSC, MIDI plugins)
    Topic: /input/delta
    """
    delta_position: Tuple[float, float, float]  # (dx, dy, dz) meters
    delta_orientation: Tuple[float, float, float]  # (drx, dry, drz) axis-angle
    source: str = "unknown"  # "keyboard", "osc", "midi", etc.
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class TargetPose:
    """
    Target end-effector pose.
    Published by: IKActor
    Topic: /control/target_pose
    """
    position: Tuple[float, float, float]  # (x, y, z) meters
    orientation: Tuple[float, float, float]  # (rx, ry, rz) axis-angle
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ControlModeChange:
    """
    Request to change control mode.
    Published by: Orchestrator
    Topic: /control/mode
    """
    mode: ControlMode
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SystemCommand:
    """
    System-level command.
    Published by: GUI or external triggers
    Topic: /system/command
    """
    command: SystemCommandType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class StateChange:
    """
    Controller state machine change.
    Published by: Orchestrator
    Topic: /system/state
    """
    state: ControllerState
    previous_state: ControllerState = ControllerState.DISCONNECTED
    message: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class UIState:
    """
    Aggregated state for UI display.
    Published by: StateActor
    Topic: /ui/state
    """
    controller_state: ControllerState
    status_message: str
    connection_healthy: bool
    joint_positions_deg: Tuple[float, ...]
    ee_position: Tuple[float, float, float]
    ee_orientation: Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    target_position: Tuple[float, float, float]
    control_mode: ControlMode
    gripper_position: float = 0.0  # Gripper position (0-255 for MuJoCo, 0-1 for real hardware)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SafetyEvent:
    """
    Safety-related event.
    Published by: SafetyActor
    Topic: /safety/event
    """
    event_type: str  # "velocity_limit", "torque_limit", "position_limit"
    details: str = ""
    severity: str = "warning"  # "warning", "error", "critical"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class EnableCommand:
    """
    Enable/disable command for actors.
    Published by: CommandRouterActor
    Topics: /control/enable, /ik/enable, /input/enable
    """
    enabled: bool
    initial_data: Dict[str, Any] = field(default_factory=dict)  # e.g., initial_q for IK
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ResetCommand:
    """
    Reset command for actors.
    Published by: CommandRouterActor
    Topic: /ik/reset
    """
    initial_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class StateTransitionRequest:
    """
    Request to transition controller state.
    Published by: CommandRouterActor, others
    Topic: /state/transition_request
    """
    target_state: ControllerState
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ModeChangeRequest:
    """
    Request to change control mode.
    Published by: CommandRouterActor
    Topic: /mode/change_request
    """
    mode: ControlMode
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ModeChanged:
    """
    Notification that mode changed.
    Published by: ModeControllerActor
    Topic: /mode/changed
    """
    mode: ControlMode
    previous_mode: ControlMode
    timestamp: float = field(default_factory=time.time)


# ============ Perception Messages (for cameras/sensors) ============

@dataclass(frozen=True, slots=True)
class ImageMessage:
    """
    Camera image data.
    Published by: PerceptionActor
    Topic: /perception/image/{camera_id}
    """
    camera_id: str
    width: int
    height: int
    channels: int
    encoding: str  # "rgb8", "bgr8", "mono8", etc.
    data: bytes  # Raw image data
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class DepthMessage:
    """
    Depth image data.
    Published by: PerceptionActor
    Topic: /perception/depth/{camera_id}
    """
    camera_id: str
    width: int
    height: int
    encoding: str  # "16UC1", "32FC1"
    data: bytes  # Raw depth data
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PointCloudMessage:
    """
    3D point cloud data.
    Published by: PerceptionActor
    Topic: /perception/pointcloud/{camera_id}
    """
    camera_id: str
    num_points: int
    points: Tuple[Tuple[float, float, float], ...]  # (x, y, z) tuples
    colors: Tuple[Tuple[int, int, int], ...] = field(default_factory=tuple)  # RGB colors
    timestamp: float = field(default_factory=time.time)


# ============ Scene/Object Messages ============

@dataclass(frozen=True, slots=True)
class ObjectState:
    """
    Dynamic state of a manipulable object in the scene.
    Published by: MuJoCoSimulator or PerceptionActor
    Topic: /scene/objects
    """
    name: str  # Object instance name (e.g., "cube1", "cube2")
    position: Tuple[float, float, float]  # (x, y, z) in meters
    orientation: Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, vz) m/s
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (wx, wy, wz) rad/s
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SceneObjects:
    """
    Collection of all object states in the scene.
    Published by: MuJoCoSimulator or PerceptionActor
    Topic: /scene/objects
    """
    objects: Dict[str, ObjectState]  # Map from object name to state
    timestamp: float = field(default_factory=time.time)


# ============ AI/ML Messages ============

@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """
    AI model inference request.
    Published by: Any actor
    Topic: /ai/inference/request
    """
    model_id: str
    input_data: Dict[str, Any]  # Flexible input (images, joint states, etc.)
    request_id: str = ""  # For tracking responses
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """
    AI model inference result.
    Published by: AIModelActor
    Topic: /ai/inference/result/{model_id}
    """
    model_id: str
    request_id: str
    output_data: Dict[str, Any]  # Flexible output
    confidence: float = 1.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
