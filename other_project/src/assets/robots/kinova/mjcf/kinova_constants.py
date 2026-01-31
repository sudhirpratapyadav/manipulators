"""Kinova Gen3 robot constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

KINOVA_GEN3_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "kinova_gen3" / "xmls" / "gen3.xml"
)
assert KINOVA_GEN3_XML.exists(), f"XML not found: {KINOVA_GEN3_XML}"

KINOVA_GEN3_TORQUE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "kinova_gen3" / "xmls" / "gen3_torque.xml"
)
assert KINOVA_GEN3_TORQUE_XML.exists(), f"XML not found: {KINOVA_GEN3_TORQUE_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load Kinova Gen3 mesh assets."""
    assets: dict[str, bytes] = {}
    # Load assets from the xmls/assets directory
    assets_path = KINOVA_GEN3_XML.parent / "assets"
    update_assets(assets, assets_path, meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3 MjSpec with assets."""
    spec = mujoco.MjSpec.from_file(str(KINOVA_GEN3_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


def get_spec_torque() -> mujoco.MjSpec:
    """Load Kinova Gen3 MjSpec for torque control (no XML actuators).

    This is used for torque control where we programmatically add motor actuators
    instead of using the position actuators.
    """
    spec = mujoco.MjSpec.from_file(str(KINOVA_GEN3_TORQUE_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Joint names.
##

ARM_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

##
# Initial state / Keyframe.
##

# Home pose - modified from gen3.xml keyframe
# qpos: 7 arm joints (no gripper on this model)
# Modified: joint_3 set to 180°, joint_4 set to +120°
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),  # Robot base on ground
    joint_pos={
        "joint_1": 0.0,
        "joint_2": 0.26179939,
        "joint_3": 3.14,  # 180° (π rad)
        "joint_4": -2.0943951,  # +120°
        "joint_5": 0.0,
        "joint_6": 0.95993109,
        "joint_7": 1.57079633,
    },
    joint_vel={".*": 0.0},
)

# Retract pose - alternative starting position
# Based on "retract" keyframe: [0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633]
RETRACT_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint_1": 0.0,
        "joint_2": -0.34906585,
        "joint_3": 3.14159265,
        "joint_4": -2.54818071,
        "joint_5": 0.0,
        "joint_6": -0.87266463,
        "joint_7": 1.57079633,
    },
    joint_vel={".*": 0.0},
)

##
# Articulation config.
##
# The gen3.xml defines position actuators for all 7 joints
# Use XmlPositionActuatorCfg to automatically use actuators defined in XML

from mjlab.actuator import BuiltinMotorActuatorCfg, XmlPositionActuatorCfg

# XmlPositionActuatorCfg automatically finds and uses actuators defined in the XML
KINOVA_ACTUATORS = XmlPositionActuatorCfg(
    joint_names_expr=(".*",),  # Match all joints
)

KINOVA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(KINOVA_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)

# Torque control actuators - uses BuiltinMotorActuatorCfg to create motor actuators
# Effort limits configured for Kinova Gen3:
# - Large actuators (joints 1-4): 32 Nm
# - Small actuators (joints 5-7): 13 Nm
KINOVA_TORQUE_ACTUATORS_LARGE = BuiltinMotorActuatorCfg(
    joint_names_expr=("|".join(ARM_JOINTS[:4]),),  # joints 1-4
    effort_limit=32.0,
    armature=0.1,
    frictionloss=1.0,
)

KINOVA_TORQUE_ACTUATORS_SMALL = BuiltinMotorActuatorCfg(
    joint_names_expr=("|".join(ARM_JOINTS[4:]),),  # joints 5-7
    effort_limit=9.0,
    armature=0.1,
    frictionloss=1.0,
)

KINOVA_TORQUE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        KINOVA_TORQUE_ACTUATORS_LARGE,
        KINOVA_TORQUE_ACTUATORS_SMALL,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_kinova_robot_cfg() -> EntityCfg:
    """Get a fresh Kinova Gen3 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),  # Use collisions from XML
        spec_fn=get_spec,
        articulation=KINOVA_ARTICULATION,
    )


def get_kinova_robot_cfg_retract() -> EntityCfg:
    """Get a fresh Kinova Gen3 robot configuration with retract pose.

    Uses retract pose instead of home pose.
    """
    return EntityCfg(
        init_state=RETRACT_INIT_STATE,
        collisions=(),  # Use collisions from XML
        spec_fn=get_spec,
        articulation=KINOVA_ARTICULATION,
    )


def get_kinova_robot_cfg_torque() -> EntityCfg:
    """Get a fresh Kinova Gen3 robot configuration with torque control.

    Uses motor actuators for direct torque control instead of position actuators.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),  # Use collisions from XML
        spec_fn=get_spec_torque,
        articulation=KINOVA_TORQUE_ARTICULATION,
    )


def get_kinova_robot_cfg_torque_retract() -> EntityCfg:
    """Get a fresh Kinova Gen3 robot configuration with torque control and retract pose.

    Uses motor actuators for direct torque control and retract pose.
    """
    return EntityCfg(
        init_state=RETRACT_INIT_STATE,
        collisions=(),  # Use collisions from XML
        spec_fn=get_spec_torque,
        articulation=KINOVA_TORQUE_ARTICULATION,
    )


# Action scale for delta control
# Using similar scale as Franka
KINOVA_ACTION_SCALE = 0.04


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_kinova_robot_cfg())

    viewer.launch(robot.spec.compile())
