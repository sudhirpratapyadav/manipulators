# Manipulators

Kinova Gen3 7-DOF robot arm control and perception system using ROS2. Features direct torque control via Kortex API, differential IK with Pinocchio, and camera-based object detection with RealSense.

## Project Structure

```
manipulators/
├── ros2_ws/                      # ROS2 workspace
│   └── src/
│       └── manipulators/         # Main ROS2 package
│           ├── src/              # Python modules (control, detection)
│           ├── scripts/          # Calibration utilities
│           ├── config/           # YAML configuration files
│           ├── launch/           # Launch files
│           ├── urdf/             # Robot description
│           └── docs/             # Diagrams and guides
└── README.md                     # This file
```

See [ros2_ws/src/manipulators/README.md](ros2_ws/src/manipulators/README.md) for detailed package documentation including:
- Architecture diagrams
- Topic/service reference
- Configuration options
- Camera calibration guide

## Features

- **Torque control** — Direct joint torque commands at 400Hz via Kortex UDP
- **Differential IK** — Task-space control with damped pseudoinverse
- **Gravity compensation** — Pinocchio-based dynamics
- **Object detection** — Pluggable detector system (HSV color, extensible to YOLO/etc)
- **Camera calibration** — Scripts for intrinsic and extrinsic calibration
- **Keyboard teleop** — Manual control for testing

## Requirements

- Docker
- VS Code with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- Kinova Gen3 7-DOF arm
- Intel RealSense camera (D435/D455)

## Dev Container Setup

This project uses a VS Code dev container with all dependencies pre-installed (ROS2 Humble, Pinocchio, Kortex API, MoveIt, cv_bridge, etc.).

### Starting the container

1. Open this folder in VS Code
2. When prompted "Reopen in Container", click **Reopen in Container**
   - Or use Command Palette (`Ctrl+Shift+P`) → "Dev Containers: Reopen in Container"
3. Wait for the container to build (first time takes a few minutes)
4. The workspace auto-builds on first start via `postCreateCommand`

### Container features

- **Base image:** `ros:humble`
- **User:** `robot` (non-root with sudo)
- **Network:** Host networking (`--net=host`) for ROS2 communication
- **Display:** X11 forwarding enabled for GUI tools (RViz, rqt)
- **ROS Domain:** `ROS_DOMAIN_ID=42`

### Installed ROS2 packages

- Core: `ros2-control`, `controller-manager`, `hardware-interface`
- Visualization: `rviz2`, `rqt`, `rqt-graph`
- Robot: `xacro`, `robot-state-publisher`, `joint-state-publisher`
- Motion: `moveit`, `joint-trajectory-controller`, `gripper-controllers`
- TF: `tf2`, `tf2-ros`, `tf2-geometry-msgs`

### References

- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Dev Containers Specification](https://containers.dev/)

## Build

After the container starts, the workspace is auto-built. To rebuild manually:

```bash
cd ~/manipulators/ros2_ws
colcon build --packages-select manipulators
source install/setup.bash
```

## Quick Start

### Control only (no camera)

```bash
# Terminal 1: Launch control node
ros2 launch manipulators diff_ik.launch.py robot_ip:=192.168.1.10

# Terminal 2: Keyboard teleop
ros2 run manipulators keyboard_teleop
```

### With object detection

```bash
# Terminal 1: Start RealSense camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# Terminal 2: Launch control node
ros2 launch manipulators diff_ik.launch.py robot_ip:=192.168.1.10

# Terminal 3: Object detection (after calibration)
ros2 run manipulators object_detection_node --ros-args \
  --params-file config/detection.yaml \
  -p camera_calibration_file:=config/camera.yaml
```

## Camera Calibration

Before using object detection, calibrate the camera:

```bash
# 1. Capture calibration images
ros2 run manipulators capture_images -- --output-dir ./calib_images --count 20

# 2. Compute intrinsics
python3 scripts/calibrate_intrinsics.py \
  --image-dir ./calib_images \
  --output config/camera.yaml \
  --board-size 8 6 \
  --square-size 25.0

# 3. Compute extrinsics (camera-to-robot transform)
ros2 run manipulators calibrate_extrinsics -- \
  --camera-yaml config/camera.yaml \
  --board-size 5 3 \
  --square-size 45.0 \
  --board-to-robot-xyz 0.0 0.0 0.0 \
  --board-to-robot-rpy 0.0 180.0 0.0
```

See [ros2_ws/src/manipulators/docs/camera_calibration.md](ros2_ws/src/manipulators/docs/camera_calibration.md) for detailed instructions.

## Environment Setup

For each new terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/manipulators/ros2_ws/install/setup.bash
```

Or add to `~/.bashrc`:

```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'source ~/manipulators/ros2_ws/install/setup.bash' >> ~/.bashrc
```

## Documentation

| Document | Description |
|----------|-------------|
| [Package README](ros2_ws/src/manipulators/README.md) | Full package documentation |
| [Diagrams](ros2_ws/src/manipulators/docs/diagrams.md) | Architecture and sequence diagrams |
| [Camera Calibration](ros2_ws/src/manipulators/docs/camera_calibration.md) | Calibration theory and guide |

## License

MIT
