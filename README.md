# Manipulators

Kinova Gen3 7-DOF robot arm control and perception system using ROS2. Features direct torque control via Kortex API, differential IK with Pinocchio, and camera-based object detection with RealSense.

## Setup Docker and Kinova
```
sudo nmcli connection add type ethernet con-name kinova-robot ifname eno1 ipv4.method manual ipv4.addresses 192.168.1.100/24
Connection 'kinova-robot' (ea9ff8ae-ee5e-40b1-8edf-cc9bbb7634d8) successfully added.
robot@robot-HP-Z2-Tower-G9-Workstation-Desktop-PC:~$ sudo nmcli connection up kinova-robot
Connection successfully activated (D-Bus active path: /org/freedesktop/NetworkManager/ActiveConnection/22)
```

```
sudo nano /etc/docker/daemon.json
```

```
{
    "bip": "192.168.100.1/24",
    "default-address-pools": [
        {
            "base": "192.168.2.0/16",
            "size": 24
        }
    ],
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```


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

### Additional Dependencies (if running outside dev container)

```bash
# xterm (required for launch files)
sudo apt install xterm

# Pinocchio (robotics kinematics/dynamics library)
pip install pin
```

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

## Kinova Kortex API Installation

If the Kortex API is not pre-installed in the container, install it manually:

```bash
# Download from Kinova Artifactory
cd /tmp
curl -L "https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.6.0/kortex_api-2.6.0.post3-py3-none-any.whl" \
  -o kortex_api-2.6.0.post3-py3-none-any.whl

# Install
pip3 install kortex_api-2.6.0.post3-py3-none-any.whl
```

### Python 3.10+ Compatibility Fix

The Kortex API requires protobuf 3.5.1, which is incompatible with Python 3.10+. Fix by upgrading protobuf:

```bash
pip3 install --force-reinstall protobuf==3.20.0
```

### Test Robot Connection

Verify connectivity to the robot:

```bash
python3 ros2_ws/src/manipulators/scripts/test_robot_connection.py --ip 192.168.1.10
```

On success, this prints the arm state and joint positions. On failure, it shows troubleshooting steps.

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
