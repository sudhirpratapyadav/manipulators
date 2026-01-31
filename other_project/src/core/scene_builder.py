"""
Scene builder for MuJoCo and Viser.

Generates MuJoCo XML scenes and provides obstacle information for Viser
based on SceneConfig.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .scene_config import SceneConfig, ObjectConfig


class SceneBuilder:
    """Builds scenes from SceneConfig for both MuJoCo and Viser."""

    def __init__(self, assets_dir: Path):
        """
        Initialize scene builder.

        Args:
            assets_dir: Path to assets directory (e.g., src/assets/)
        """
        self.assets_dir = Path(assets_dir)

    def build_mujoco_scene(self, scene: SceneConfig, output_path: str) -> str:
        """
        Build a MuJoCo XML scene file from SceneConfig.

        Args:
            scene: Scene configuration
            output_path: Path to save the generated XML file

        Returns:
            Path to the generated scene file
        """
        # Get robot MJCF path (use version without keyframes for scene composition)
        robot_mjcf = self.assets_dir / "robots" / scene.robot.name / "mjcf" / "gen3_no_keyframes.xml"
        if not robot_mjcf.exists():
            raise FileNotFoundError(f"Robot MJCF not found: {robot_mjcf}")

        # Create root mujoco element
        mujoco = ET.Element("mujoco", model="scene")

        # Include robot MJCF at top level (MuJoCo requires includes before other elements)
        include = ET.SubElement(mujoco, "include", file=str(robot_mjcf.resolve()))

        # Add visual settings
        visual = ET.SubElement(mujoco, "visual")
        ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
        headlight = ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.8 0.8 0.8")

        # Add assets for obstacles
        asset = ET.SubElement(mujoco, "asset")

        # Asset configuration: mesh files and default materials per asset type
        ASSET_CONFIG = {
            "table": {
                "mesh_file": "table.stl",
                "mesh_scale": "1 1 0.8",
                "mesh_dir": "objects/table/meshes",
            }
        }

        # Add meshes and materials for all objects
        for obj in scene.objects:
            # Add mesh asset if configured
            if obj.asset in ASSET_CONFIG and "mesh_file" in ASSET_CONFIG[obj.asset]:
                config = ASSET_CONFIG[obj.asset]
                mesh_path = self.assets_dir / config["mesh_dir"] / config["mesh_file"]

                if mesh_path.exists():
                    ET.SubElement(asset, "mesh",
                                 name=f"{obj.name}_mesh",
                                 file=str(mesh_path.resolve()),
                                 scale=config["mesh_scale"])

            # Add material
            if obj.color:
                r, g, b, a = obj.color
                ET.SubElement(asset, "material",
                             name=f"mat_{obj.name}",
                             rgba=f"{r} {g} {b} {a}")

        # Add worldbody
        worldbody = ET.SubElement(mujoco, "worldbody")

        # Add lighting
        ET.SubElement(worldbody, "light",
                     name="light1",
                     pos="1 1 2",
                     dir="-1 -1 -2",
                     diffuse="0.8 0.8 0.8")

        # Add ground plane at configured height
        ground_geom = ET.SubElement(worldbody, "geom",
                     name="floor",
                     type="plane",
                     size="2 2 0.1",
                     rgba="0.9 0.9 0.9 1",
                     contype="1",
                     conaffinity="1")
        # Set ground position (only z matters for infinite plane)
        ground_geom.set("pos", f"0 0 {scene.ground_height}")

        # Note: Robot base pose will be set in keyframe qpos, not as a body transform
        # MuJoCo include merges the robot's bodies directly into worldbody

        # Add all objects (static and dynamic)
        dynamic_objects = []
        for obj in scene.objects:
            if obj.static:
                self._add_static_object_to_worldbody(worldbody, obj)
            else:
                self._add_dynamic_object_to_worldbody(worldbody, obj)
                dynamic_objects.append(obj)

        # Note: Keyframes are tricky with dynamic objects
        # The included robot XML has keyframes, but they don't account for dynamic object DOFs
        # For now, we'll let MuJoCo use the included keyframes (they only affect robot joints)
        # Dynamic objects will start at their body-defined initial poses

        # Pretty print and save
        xml_str = self._prettify_xml(mujoco)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            f.write(xml_str)

        print(f"[SceneBuilder] Generated MuJoCo scene: {output_file}")
        return str(output_file)

    def _add_static_object_to_worldbody(self, worldbody: ET.Element, obj: ObjectConfig):
        """Add a static object to the worldbody."""
        # Create body for static object
        body = ET.SubElement(worldbody, "body", name=obj.name)

        # Set position
        pos = obj.pose.position
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")

        # Set orientation (quaternion)
        quat = obj.pose.orientation  # [x, y, z, w]
        body.set("quat", f"{quat[3]} {quat[0]} {quat[1]} {quat[2]}")  # MuJoCo uses [w, x, y, z]

        # Add geometry based on asset type
        if obj.asset == "cuboid":
            self._add_cuboid_geom(body, obj)
        elif obj.asset == "table":
            self._add_table_geom(body, obj)
        else:
            print(f"[SceneBuilder] Warning: Unknown asset type '{obj.asset}' for object '{obj.name}'")

    def _add_dynamic_object_to_worldbody(self, worldbody: ET.Element, obj: ObjectConfig):
        """
        Add a dynamic object to the worldbody as a free body.

        Dynamic objects have freejoints so they can move in simulation.
        """
        # Create body for dynamic object with freejoint
        body = ET.SubElement(worldbody, "body", name=obj.name)

        # Set initial position
        pos = obj.pose.position
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")

        # Set initial orientation (quaternion)
        quat = obj.pose.orientation  # [x, y, z, w]
        body.set("quat", f"{quat[3]} {quat[0]} {quat[1]} {quat[2]}")  # MuJoCo uses [w, x, y, z]

        # Add freejoint (allows 6DOF movement)
        ET.SubElement(body, "freejoint", name=f"{obj.name}_joint")

        # Add geometry based on asset type
        if obj.asset == "cuboid":
            self._add_cuboid_geom_for_dynamic(body, obj)
        else:
            print(f"[SceneBuilder] Warning: Unknown asset type '{obj.asset}' for dynamic object '{obj.name}'")

    def _add_cuboid_geom(self, body: ET.Element, obj: ObjectConfig):
        """Add a box geometry to a body (for static objects)."""
        # Get size (MuJoCo box size is half-extents)
        size = obj.size if obj.size else [0.1, 0.1, 0.1]
        half_size = [s / 2.0 for s in size]

        # Create geom
        geom_attribs = {
            "name": f"{obj.name}_geom",
            "type": "box",
            "size": f"{half_size[0]} {half_size[1]} {half_size[2]}",
            "contype": "1",
            "conaffinity": "1",
        }

        # Add material/color
        if obj.color:
            geom_attribs["material"] = f"mat_{obj.name}"
        else:
            geom_attribs["rgba"] = "0.8 0.8 0.8 1.0"

        ET.SubElement(body, "geom", **geom_attribs)

    def _add_table_geom(self, body: ET.Element, obj: ObjectConfig):
        """
        Add table geometry with mesh.

        Config matches table.xml (single ground truth):
        - Mesh: table.stl with scale 1 1 0.8
        - Rotation: 90 degrees around Z-axis
        - Size: 0.8m x 1.2m x 0.68m (width x depth x height including legs)
        """
        # Default table dimensions (from table.xml)
        size = obj.size if obj.size else [0.8, 1.2, 0.68]
        half_size = [s / 2.0 for s in size]

        # Visual geom with mesh
        visual_attribs = {
            "name": f"{obj.name}_visual",
            "type": "mesh",
            "mesh": f"{obj.name}_mesh",
            "contype": "0",
            "conaffinity": "0",
            "mass": "5.0",
            "euler": "0 0 1.57",  # 90-degree rotation (matches table.xml)
        }
        if obj.color:
            visual_attribs["material"] = f"mat_{obj.name}"
        else:
            visual_attribs["rgba"] = "0.8 0.7 0.6 1.0"

        ET.SubElement(body, "geom", **visual_attribs)

        # Collision geom (invisible box for physics)
        # Position the box center at half height so it sits above the ground
        collision_attribs = {
            "name": f"{obj.name}_collision",
            "type": "box",
            "size": f"{half_size[0]} {half_size[1]} {half_size[2]}",
            "pos": f"0 0 {half_size[2]}",  # Offset by half-height in Z
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0 0 0 0",
            "euler": "0 0 1.57",  # 90-degree rotation (matches table.xml)
        }

        ET.SubElement(body, "geom", **collision_attribs)

    def _add_cuboid_geom_for_dynamic(self, body: ET.Element, obj: ObjectConfig):
        """Add a box geometry to a free body (for dynamic objects)."""
        # Get size (MuJoCo box size is half-extents)
        size = obj.size if obj.size else [0.05, 0.05, 0.05]
        half_size = [s / 2.0 for s in size]

        # Create geom with mass for dynamics
        geom_attribs = {
            "name": f"{obj.name}_geom",
            "type": "box",
            "size": f"{half_size[0]} {half_size[1]} {half_size[2]}",
            "contype": "1",
            "conaffinity": "1",
            "mass": "0.1",  # 100g default mass
        }

        # Add material/color
        if obj.color:
            geom_attribs["material"] = f"mat_{obj.name}"
        else:
            geom_attribs["rgba"] = "0.5 0.5 0.5 1.0"

        ET.SubElement(body, "geom", **geom_attribs)

    def _add_keyframes_with_objects(self, mujoco: ET.Element, dynamic_objects: List):
        """
        Add keyframes with extended qpos to include dynamic objects.

        The robot XML includes keyframes for 7 arm joints + 8 gripper joints (15 values).
        With dynamic objects (each has 7 DOF freejoint), we need to extend the qpos.
        """
        # Robot home keyframe qpos (7 arm + 8 gripper = 15 values)
        robot_home_qpos = "0 0.26179939 3.14159265 -2.26892803 0 0.95993109 1.57079633 0 0 0 0 0 0 0 0"
        robot_home_ctrl = "0 0.26179939 3.14159265 -2.26892803 0 0.95993109 1.57079633 0"

        # For each dynamic object, add its initial pose (3 pos + 4 quat = 7 values)
        object_qpos_parts = []
        for obj in dynamic_objects:
            pos = obj.pose.position
            quat = obj.pose.orientation  # [x, y, z, w]
            # Convert to MuJoCo format [w, x, y, z]
            quat_mujoco = [quat[3], quat[0], quat[1], quat[2]]
            obj_qpos = f"{pos[0]} {pos[1]} {pos[2]} {quat_mujoco[0]} {quat_mujoco[1]} {quat_mujoco[2]} {quat_mujoco[3]}"
            object_qpos_parts.append(obj_qpos)

        # Combine robot and object qpos
        full_qpos = robot_home_qpos + " " + " ".join(object_qpos_parts)

        # Create keyframe element (overrides the included one)
        keyframe = ET.SubElement(mujoco, "keyframe")
        ET.SubElement(keyframe, "key",
                     name="home",
                     qpos=full_qpos,
                     ctrl=robot_home_ctrl)

    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def get_ground_height(self, scene: SceneConfig) -> float:
        """
        Get ground plane height from scene configuration.

        Args:
            scene: Scene configuration

        Returns:
            Ground height in meters
        """
        return scene.ground_height

    def get_scene_objects(self, scene: SceneConfig) -> List[dict]:
        """
        Get all scene objects (static and dynamic) for Viser visualization.

        Args:
            scene: Scene configuration

        Returns:
            List of object info dicts with name, type, pose, size, color, static flag, urdf_path (if available)
        """
        objects = []

        for obj in scene.objects:
            obj_info = {
                "name": obj.name,
                "type": obj.asset,
                "position": np.array(obj.pose.position),
                "orientation": np.array(obj.pose.orientation),  # quaternion [x, y, z, w]
                "static": obj.static,  # True for obstacles, False for manipulable objects
            }

            if obj.size:
                obj_info["size"] = np.array(obj.size)

            if obj.color:
                obj_info["color"] = tuple(obj.color[:3])  # RGB only for Viser
            else:
                # Default colors: lighter for static, darker for dynamic
                obj_info["color"] = (0.8, 0.8, 0.8) if obj.static else (0.5, 0.5, 0.5)

            # Check if URDF exists for this object (for mesh loading in Viser)
            urdf_path = self.assets_dir / "objects" / obj.asset / "urdf" / f"{obj.asset}.urdf"
            if urdf_path.exists():
                obj_info["urdf_path"] = str(urdf_path)

            objects.append(obj_info)

        return objects
