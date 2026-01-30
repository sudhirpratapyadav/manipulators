# Manipulators

ROS2 package for Kinova Gen3 7-DOF torque control using Pinocchio for differential IK and gravity compensation. Communicates directly with the robot via Kortex API (no ros2_control drivers).

## Architecture

```
keyboard_teleop ──/target_pose──> control_node ──torques──> Kinova (UDP @ 400Hz)
                  /gripper_cmd──┘      |
                                       |── /joint_states
                                       └── /ee_pose
```

The control node runs a single controller (defined by launch file), with a fixed lifecycle:

```
Startup:  connect → clear faults → home → low-level servoing → torque mode → control loop
Shutdown: stop loop → position mode → high-level servoing → home → disconnect
```

## Package Structure

```
src/
├── hardware.py              # KinovaHardware — Kortex TCP/UDP I/O + gripper
├── robot_model.py           # Pinocchio wrapper — gravity, Jacobian, FK (7-DOF reduced model)
├── diff_ik_controller.py    # Diff-IK + damped pseudoinverse + gravity comp → joint torques
├── control_node.py          # Main ROS2 node — startup/loop/shutdown
├── keyboard_teleop.py       # Keyboard → /target_pose + /gripper_command
└── utility.py               # Quaternion, rotation, pose error helpers
```

## Usage

```bash
# Build
cd ~/manipulators/ros2_ws
colcon build --packages-select manipulators
source install/setup.bash

# Run with diff-IK + keyboard teleop
ros2 launch manipulators diff_ik.launch.py robot_ip:=192.168.1.10

# Emergency stop (from another terminal)
ros2 service call /e_stop std_srvs/srv/Trigger
```

## Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/target_pose` | `geometry_msgs/PoseStamped` | Input | Desired EE pose |
| `/gripper_command` | `std_msgs/Float64` | Input | Gripper target (0.0=open, 1.0=closed) |
| `/joint_states` | `sensor_msgs/JointState` | Output | Joint positions, velocities, torques |
| `/ee_pose` | `geometry_msgs/PoseStamped` | Output | Current EE pose |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/e_stop` | `std_srvs/Trigger` | Emergency stop — kills torque mode immediately |

## Configuration

All parameters in `config/kinova_gen3.yaml`:

- `robot_ip` — Kinova IP address
- `home_position_deg` — Joint angles for home position (degrees, Kinova 0-360 convention)
- `control_rate_hz` — Control loop frequency (default 400 Hz)
- `kp_task` — Task-space proportional gains `[x, y, z, rx, ry, rz]`
- `kp_joint` — Joint-space position gains (per joint, default 0.0 = disabled)
- `kd_joint` — Joint-space damping gains (per joint)
- `damping` — Pseudoinverse regularization
- `max_joint_velocity` — Safety clamp on diff-IK output (rad/s)
- `max_torque` — Per-joint torque limits (Nm)

## Keyboard Teleop Keys

```
W/S  — X forward/back        U/O — roll +/-
A/D  — Y left/right          I/K — pitch +/-
Q/E  — Z up/down             J/L — yaw +/-
G    — toggle gripper         ESC — quit
```

## Adding a New Controller

1. Create `src/new_controller.py` with a `compute(target_pos, target_quat, q, dq) -> torques` method
2. Import and instantiate it in a copy of `control_node.py` (or parameterize the existing one)
3. Add a new launch file `launch/new_controller.launch.py`

## Diagrams

See [docs/diagrams.md](docs/diagrams.md) for detailed mermaid diagrams covering:

- System architecture
- Startup / shutdown sequences
- Control loop (single cycle breakdown)
- E-stop sequence
- Threading model
- Module dependency graph
- Data flow and coordinate conversions
- Kinova hardware connection architecture (TCP vs UDP)

## Dependencies

- ROS2 (Humble+)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) — dynamics, Jacobians, FK
- [Kortex API](https://github.com/Kinovarobotics/kortex) — direct robot communication
- NumPy
