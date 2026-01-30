# Camera Calibration Guide

This guide explains how to calibrate your RealSense camera for use with the object detection system. Calibration produces the parameters needed to accurately project 2D pixel coordinates to 3D positions in the robot's coordinate frame.

## Why Calibration?

Camera calibration solves two problems:

1. **Intrinsic calibration** — corrects lens distortion and determines how the camera maps 3D points to 2D pixels
2. **Extrinsic calibration** — determines where the camera is located relative to the robot base

Without proper calibration, detected objects will appear at incorrect 3D positions.

---

## Quick Start

```bash
# 1. Start the RealSense camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 2. Capture calibration images
ros2 run manipulators capture_images -- --output-dir ./calib_images --count 20

# 3. Compute intrinsics
python3 scripts/calibrate_intrinsics.py \
  --image-dir ./calib_images \
  --output config/camera.yaml \
  --board-size 8 6 \
  --square-size 25.0

# 4. Compute extrinsics
ros2 run manipulators calibrate_extrinsics -- \
  --camera-yaml config/camera.yaml \
  --board-size 5 3 \
  --square-size 45.0 \
  --board-to-robot-xyz -0.11 -0.01 0.0 \
  --board-to-robot-rpy 0.0 180.0 0.0
```

---

## Prerequisites

- A chessboard calibration pattern (printed on flat, rigid surface)
- The RealSense camera mounted in its final position (eye-to-hand, fixed)
- The `realsense2_camera` ROS2 driver installed

### Chessboard sizing

The `--board-size` parameter specifies the number of **inner corners**, not squares:

```
8x6 chessboard pattern:
┌─┬─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │ │ │ │ │ │  ← 9 columns of squares
├─┼─┼─┼─┼─┼─┼─┼─┼─┤      = 8 inner corner columns
│ │●│●│●│●│●│●│●│ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┤  ← inner corners (●)
│ │●│●│●│●│●│●│●│ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │●│●│●│●│●│●│●│ │      7 rows of squares
├─┼─┼─┼─┼─┼─┼─┼─┼─┤      = 6 inner corner rows
│ │●│●│●│●│●│●│●│ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │●│●│●│●│●│●│●│ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┴─┘
```

So for this 9x7 square grid, use `--board-size 8 6`.

---

## Step 1: Capture Calibration Images

Start the camera driver:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

Run the capture script:

```bash
ros2 run manipulators capture_images -- \
  --output-dir ./calib_images \
  --count 20 \
  --topic /camera/camera/color/image_raw
```

**Controls:**
- `SPACE` — capture current frame
- `ESC` — quit early

**Tips for good calibration images:**

- Capture 15-25 images
- Move the chessboard to different positions and angles
- Cover the entire field of view (corners and edges, not just center)
- Tilt the board at various angles (up to ~45°)
- Ensure the chessboard is fully visible in every capture
- Avoid motion blur — hold still when capturing

Bad coverage (all images in center):
```
┌─────────────────┐
│                 │
│     ███████     │
│     ███████     │  ← poor
│     ███████     │
│                 │
└─────────────────┘
```

Good coverage (spread across frame):
```
┌─────────────────┐
│ ██     ██    ██ │
│    ██      ██   │
│ ██    ██     ██ │  ← good
│    ██     ██    │
│ ██     ██    ██ │
└─────────────────┘
```

---

## Step 2: Intrinsic Calibration

Compute the camera matrix and distortion coefficients:

```bash
python3 scripts/calibrate_intrinsics.py \
  --image-dir ./calib_images \
  --output config/camera.yaml \
  --board-size 8 6 \
  --square-size 25.0
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--image-dir` | Directory containing captured PNG images |
| `--output` | Output YAML file path |
| `--board-size` | Inner corners (cols rows), e.g., `8 6` |
| `--square-size` | Size of one square in millimeters |

**Output:**

The script writes the following to `camera.yaml`:

```yaml
intrinsics:
  camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]  # 3x3 flattened
  dist_coeffs: [k1, k2, p1, p2, k3]               # distortion
  image_size: [width, height]
  reprojection_error: 0.xx                        # should be < 0.5
```

### Understanding the results

- **camera_matrix** contains focal lengths (fx, fy) and principal point (cx, cy)
- **dist_coeffs** contains radial (k1, k2, k3) and tangential (p1, p2) distortion
- **reprojection_error** — how well the model fits; should be < 0.5 pixels, ideally < 0.3

If reprojection error is high (> 1.0):
- Re-capture images with better coverage/angles
- Check that the printed pattern is flat (not warped)
- Ensure correct `--board-size` and `--square-size`

---

## Step 3: Extrinsic Calibration

This step determines where the camera is in the robot's coordinate frame.

### Setup

1. Place a chessboard at a **known position** relative to the robot base
2. The chessboard should be clearly visible to the camera
3. Measure the chessboard origin's position and orientation relative to robot base

### Coordinate system convention

```
Robot base frame:         Chessboard frame:
      Z                         Z (normal)
      │                         │
      │                         │
      └────── Y                 └────── X
     /                         /
    X                         Y
```

The chessboard origin is at the **first inner corner** (top-left when facing the board).

### Running extrinsic calibration

```bash
ros2 run manipulators calibrate_extrinsics -- \
  --camera-yaml config/camera.yaml \
  --board-size 5 3 \
  --square-size 45.0 \
  --board-to-robot-xyz -0.11 -0.01 0.0 \
  --board-to-robot-rpy 0.0 180.0 0.0
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--camera-yaml` | Path to camera.yaml with intrinsics |
| `--board-size` | Inner corners of the extrinsic board |
| `--square-size` | Square size in mm |
| `--board-to-robot-xyz` | Translation from board origin to robot base (meters) |
| `--board-to-robot-rpy` | Rotation from board to robot (degrees, roll-pitch-yaw) |

### How to measure board-to-robot transform

If you place the chessboard flat on the table with the robot at the origin:

```
Top view:

    ┌─────────┐   Robot base at origin (0, 0, 0)
    │ board   │   Board origin at (-0.4, 0.2, 0.0) meters
    │ ○───────│   ○ = first inner corner
    └─────────┘
         ↑
         └── measure this offset

              ●────→ Y (robot)
              │
              ↓
              X (robot)
```

For rotation:
- If the board's X-axis points in the same direction as robot's X-axis: RPY = (0, 0, 0)
- If the board is flipped (facing up): RPY = (0, 180, 0)
- If the board is rotated 90° on the table: RPY = (0, 0, 90)

### Interactive mode

Running without `--board-to-robot-*` arguments enters interactive mode:

```bash
ros2 run manipulators calibrate_extrinsics -- \
  --camera-yaml config/camera.yaml \
  --board-size 5 3 \
  --square-size 45.0
```

This lets you:
- Press `c` to capture and display detected corners
- Enter the transform interactively
- Press `s` to save when satisfied

**Output:**

Appends to `camera.yaml`:

```yaml
extrinsics:
  rvec: [rx, ry, rz]           # Rodrigues rotation vector (camera → robot)
  tvec: [tx, ty, tz]           # Translation vector (camera → robot)
  camera_to_robot_matrix: [...]  # 4x4 transform, flattened
```

---

## Verifying Calibration

### Visual check

Run the object detection node and verify that detected 3D points match reality:

```bash
# Terminal 1: camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# Terminal 2: detection node
ros2 run manipulators object_detection_node --ros-args \
  --params-file config/detection.yaml \
  -p camera_calibration_file:=config/camera.yaml

# Terminal 3: monitor detections
ros2 topic echo /detected_object_point
```

Place an object at a known position (e.g., 0.4m in front of robot, 0.1m to the left) and check if the detected coordinates match.

### Common issues

| Problem | Likely cause | Solution |
|---------|--------------|----------|
| Large position offset | Wrong extrinsic transform | Re-measure board-to-robot pose |
| Systematic X/Y error | Board rotation wrong | Check RPY angles |
| Z offset | Table height not accounted for | Include Z in board-to-robot-xyz |
| Random scatter | Poor intrinsics | Re-capture with better coverage |

---

## Output File Format

After both calibrations, `config/camera.yaml` contains:

```yaml
intrinsics:
  camera_matrix: [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
  dist_coeffs: [0.1, -0.25, 0.0, 0.0, 0.1]
  image_size: [640, 480]
  reprojection_error: 0.32

extrinsics:
  rvec: [2.1, 0.5, -0.3]
  tvec: [0.2, -0.1, 1.5]
  camera_to_robot_matrix: [r11, r12, r13, tx, r21, ..., 0, 0, 0, 1]
```

The `object_detection_node` loads this file and uses it to:
1. Undistort the image (optional, using intrinsics)
2. Project 2D detections to 3D using depth + intrinsics
3. Transform camera-frame 3D points to robot frame using extrinsics

---

## Tips

- **RealSense factory intrinsics**: The RealSense camera publishes factory calibration on `/camera_info`. You can skip intrinsic calibration and use these directly by leaving `intrinsics` empty in `camera.yaml` — the node will fall back to `/camera_info`.

- **Re-calibrate extrinsics if camera moves**: Intrinsics only change if you swap lenses or cameras. Extrinsics must be re-done whenever the camera is repositioned.

- **Multiple calibration boards**: You can use different boards for intrinsics vs extrinsics. Intrinsics benefits from many small squares; extrinsics just needs a board you can position accurately.

---

## Troubleshooting

**"No chessboard detected in any image"**
- Check `--board-size` matches your actual pattern (inner corners, not squares)
- Ensure images are not too dark or motion-blurred
- Print pattern at larger size or move camera closer

**High reprojection error (> 1.0)**
- Images may have motion blur
- Board may be warped or not flat
- Wrong square size specified

**Detection positions are offset by a constant**
- Extrinsic transform is wrong
- Double-check board-to-robot measurements

**Detection positions are rotated**
- RPY angles incorrect
- Check which direction the chessboard X-axis points
