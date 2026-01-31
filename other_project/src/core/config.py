"""
Configuration management with YAML loading and validation.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class KinovaConfig:
    """Kinova robot connection settings."""
    ip: str = "192.168.1.10"
    username: str = "admin"
    password: str = "admin"


@dataclass
class GainsConfig:
    """PD controller gains."""
    kp: List[float] = field(default_factory=lambda: [120.0, 120.0, 120.0, 120.0, 80.0, 80.0, 80.0])
    kd: List[float] = field(default_factory=lambda: [12.0, 12.0, 12.0, 12.0, 8.0, 8.0, 2.0])


@dataclass
class LimitsConfig:
    """Safety limits."""
    max_velocity_deg_s: float = 10000.0  # Very high for simulation (was 50.0)
    max_pd_torque_nm: float = 200.0  # Very high for simulation (was 10.0)
    position_bound_m: float = 0.1  # Large workspace for simulation (was 0.1)


@dataclass
class RatesConfig:
    """Loop rates for different actors."""
    hardware_hz: int = 1000
    ik_hz: int = 400
    gui_hz: int = 50
    safety_hz: int = 1000


@dataclass
class IKConfig:
    """Differential IK parameters."""
    lambda_min: float = 0.01
    lambda_max: float = 0.5
    manip_threshold: float = 0.01
    max_step_rad: float = 0.2


@dataclass
class KeyboardConfig:
    """Keyboard input settings."""
    position_step_m: float = 0.05
    rotation_step_rad: float = 0.1


@dataclass
class InputsConfig:
    """Input sources configuration."""
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)


@dataclass
class PerceptionConfig:
    """Perception settings."""
    camera_id: str = "realsense"


@dataclass
class SimulationConfig:
    """Simulation settings."""
    enabled: bool = False
    model_path: str = "src/assets/robots/kinova/mjcf/scene.xml"
    render: bool = True
    use_torque_actuators: bool = False  # True for gen3_torque.xml, False for gen3.xml
    scene_config: str = "src/config/pick_place_scene.yaml"  # Scene configuration file


@dataclass
class ControlConfig:
    """Control system configuration."""
    rates: RatesConfig = field(default_factory=RatesConfig)
    gains: GainsConfig = field(default_factory=GainsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    ik: IKConfig = field(default_factory=IKConfig)


@dataclass
class Config:
    """Root configuration."""
    kinova: KinovaConfig = field(default_factory=KinovaConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    inputs: InputsConfig = field(default_factory=InputsConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    home_joints_rad: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.home_joints_rad:
            # Default home position
            self.home_joints_rad = [-0.217, 0.993, -2.821, -1.434, -0.383, -0.783, 1.914]


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Config object with all settings.
    """
    if config_path is None:
        return Config()

    path = Path(config_path)
    if not path.exists():
        print(f"[Config] Warning: {config_path} not found, using defaults")
        return Config()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return _dict_to_config(data)


def _dict_to_config(data: dict) -> Config:
    """Convert dictionary to Config object."""
    config = Config()

    # Kinova
    if "kinova" in data:
        k = data["kinova"]
        config.kinova = KinovaConfig(
            ip=k.get("ip", config.kinova.ip),
            username=k.get("username", config.kinova.username),
            password=k.get("password", config.kinova.password),
        )

    # Control
    if "control" in data:
        c = data["control"]

        if "rates" in c:
            r = c["rates"]
            config.control.rates = RatesConfig(
                hardware_hz=r.get("hardware_hz", config.control.rates.hardware_hz),
                ik_hz=r.get("ik_hz", config.control.rates.ik_hz),
                gui_hz=r.get("gui_hz", config.control.rates.gui_hz),
                safety_hz=r.get("safety_hz", config.control.rates.safety_hz),
            )

        if "gains" in c:
            g = c["gains"]
            config.control.gains = GainsConfig(
                kp=g.get("kp", config.control.gains.kp),
                kd=g.get("kd", config.control.gains.kd),
            )

        if "limits" in c:
            l = c["limits"]
            config.control.limits = LimitsConfig(
                max_velocity_deg_s=l.get("max_velocity_deg_s", config.control.limits.max_velocity_deg_s),
                max_pd_torque_nm=l.get("max_pd_torque_nm", config.control.limits.max_pd_torque_nm),
                position_bound_m=l.get("position_bound_m", config.control.limits.position_bound_m),
            )

        if "ik" in c:
            ik = c["ik"]
            config.control.ik = IKConfig(
                lambda_min=ik.get("lambda_min", config.control.ik.lambda_min),
                lambda_max=ik.get("lambda_max", config.control.ik.lambda_max),
                manip_threshold=ik.get("manip_threshold", config.control.ik.manip_threshold),
                max_step_rad=ik.get("max_step_rad", config.control.ik.max_step_rad),
            )

    # Inputs
    if "inputs" in data:
        i = data["inputs"]

        if "keyboard" in i:
            kb = i["keyboard"]
            config.inputs.keyboard = KeyboardConfig(
                position_step_m=kb.get("position_step_m", config.inputs.keyboard.position_step_m),
                rotation_step_rad=kb.get("rotation_step_rad", config.inputs.keyboard.rotation_step_rad),
            )

    # Perception
    if "perception" in data:
        p = data["perception"]
        config.perception = PerceptionConfig(
            camera_id=p.get("camera_id", config.perception.camera_id),
        )

    # Simulation
    if "simulation" in data:
        s = data["simulation"]
        config.simulation = SimulationConfig(
            enabled=s.get("enabled", config.simulation.enabled),
            model_path=s.get("model_path", config.simulation.model_path),
            render=s.get("render", config.simulation.render),
            use_torque_actuators=s.get("use_torque_actuators", config.simulation.use_torque_actuators),
            scene_config=s.get("scene_config", config.simulation.scene_config),
        )

    # Home position
    if "home_joints_rad" in data:
        config.home_joints_rad = data["home_joints_rad"]

    return config


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file."""
    data = {
        "kinova": {
            "ip": config.kinova.ip,
            "username": config.kinova.username,
            "password": config.kinova.password,
        },
        "control": {
            "rates": {
                "hardware_hz": config.control.rates.hardware_hz,
                "ik_hz": config.control.rates.ik_hz,
                "gui_hz": config.control.rates.gui_hz,
                "safety_hz": config.control.rates.safety_hz,
            },
            "gains": {
                "kp": config.control.gains.kp,
                "kd": config.control.gains.kd,
            },
            "limits": {
                "max_velocity_deg_s": config.control.limits.max_velocity_deg_s,
                "max_pd_torque_nm": config.control.limits.max_pd_torque_nm,
                "position_bound_m": config.control.limits.position_bound_m,
            },
            "ik": {
                "lambda_min": config.control.ik.lambda_min,
                "lambda_max": config.control.ik.lambda_max,
                "manip_threshold": config.control.ik.manip_threshold,
                "max_step_rad": config.control.ik.max_step_rad,
            },
        },
        "inputs": {
            "keyboard": {
                "position_step_m": config.inputs.keyboard.position_step_m,
                "rotation_step_rad": config.inputs.keyboard.rotation_step_rad,
            },
        },
        "perception": {
            "camera_id": config.perception.camera_id,
        },
        "simulation": {
            "enabled": config.simulation.enabled,
            "model_path": config.simulation.model_path,
            "render": config.simulation.render,
            "use_torque_actuators": config.simulation.use_torque_actuators,
            "scene_config": config.simulation.scene_config,
        },
        "home_joints_rad": config.home_joints_rad,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
