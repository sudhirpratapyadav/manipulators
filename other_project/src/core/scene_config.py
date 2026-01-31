"""
Scene configuration schema for robot environments.

Defines the robot pose, joint configuration, and obstacles in the scene.
This config is used by both simulation (MuJoCo) and visualization (Viser).
"""

import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class Pose:
    """6-DOF pose (position + orientation)."""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # xyz in meters
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # quaternion (x, y, z, w)

    def to_position_array(self) -> np.ndarray:
        """Get position as numpy array."""
        return np.array(self.position)

    def to_orientation_array(self) -> np.ndarray:
        """Get orientation quaternion as numpy array."""
        return np.array(self.orientation)


@dataclass
class RobotConfig:
    """Robot configuration in the scene."""
    name: str = "kinova"  # Robot type (maps to assets/robots/{name}/)
    base_pose: Pose = field(default_factory=Pose)
    joint_config: List[float] = field(default_factory=list)  # Joint angles in radians
    gripper_position: float = 0.0  # Gripper position 0-1 normalized

    def __post_init__(self):
        if not self.joint_config:
            # Default home position for Kinova
            self.joint_config = [-0.217, 0.993, -2.821, -1.434, -0.383, -0.783, 1.914]


@dataclass
class ObjectConfig:
    """
    Object configuration (can be static or dynamic).

    - static=True: Fixed obstacle (e.g., table, wall) - no state updates
    - static=False: Manipulable object with dynamic state from sim/perception
    """
    name: str  # Instance name (e.g., "table", "cube1")
    asset: str  # Asset type (maps to assets/objects/{asset}/)
    pose: Pose = field(default_factory=Pose)  # Pose for static, initial_pose for dynamic
    static: bool = True  # True for obstacles, False for manipulable objects
    params: Dict[str, Any] = field(default_factory=dict)  # Asset-specific parameters

    @property
    def size(self) -> Optional[List[float]]:
        """Get size parameter if it exists."""
        return self.params.get("size")

    @property
    def color(self) -> Optional[List[float]]:
        """Get color parameter if it exists."""
        return self.params.get("color")


@dataclass
class SceneConfig:
    """Complete scene configuration."""
    robot: RobotConfig = field(default_factory=RobotConfig)
    objects: List[ObjectConfig] = field(default_factory=list)  # All objects (static + dynamic)
    ground_height: float = 0.0  # Ground plane z-coordinate in meters


def load_scene_config(config_path: str) -> SceneConfig:
    """
    Load scene configuration from YAML file.

    Args:
        config_path: Path to scene config file

    Returns:
        SceneConfig object
    """
    path = Path(config_path)
    if not path.exists():
        print(f"[SceneConfig] Warning: {config_path} not found, using defaults")
        return SceneConfig()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return _dict_to_scene_config(data)


def _dict_to_scene_config(data: dict) -> SceneConfig:
    """Convert dictionary to SceneConfig object."""
    scene = SceneConfig()

    # Ground height
    scene.ground_height = data.get("ground_height", 0.0)

    # Robot configuration
    if "robot" in data:
        r = data["robot"]
        robot_name = r.get("name", "kinova")

        # Parse base pose
        base_pose = Pose()
        if "base_pose" in r:
            bp = r["base_pose"]
            base_pose.position = bp.get("position", base_pose.position)
            base_pose.orientation = bp.get("orientation", base_pose.orientation)

        scene.robot = RobotConfig(
            name=robot_name,
            base_pose=base_pose,
            joint_config=r.get("joint_config", []),
            gripper_position=r.get("gripper_position", 0.0)
        )

    # Objects (both static and dynamic)
    if "objects" in data:
        for obj_data in data["objects"]:
            # Parse pose
            pose = Pose()
            if "pose" in obj_data:
                p = obj_data["pose"]
                pose.position = p.get("position", pose.position)
                pose.orientation = p.get("orientation", pose.orientation)

            obj = ObjectConfig(
                name=obj_data["name"],
                asset=obj_data["asset"],
                pose=pose,
                static=obj_data.get("static", True),  # Default to static
                params=obj_data.get("params", {})
            )
            scene.objects.append(obj)

    return scene


def save_scene_config(scene: SceneConfig, path: str) -> None:
    """Save scene configuration to YAML file."""
    data = {
        "ground_height": scene.ground_height,
        "robot": {
            "name": scene.robot.name,
            "base_pose": {
                "position": scene.robot.base_pose.position,
                "orientation": scene.robot.base_pose.orientation,
            },
            "joint_config": scene.robot.joint_config,
            "gripper_position": scene.robot.gripper_position,
        },
        "objects": [
            {
                "name": obj.name,
                "asset": obj.asset,
                "pose": {
                    "position": obj.pose.position,
                    "orientation": obj.pose.orientation,
                },
                "static": obj.static,
                "params": obj.params,
            }
            for obj in scene.objects
        ]
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
