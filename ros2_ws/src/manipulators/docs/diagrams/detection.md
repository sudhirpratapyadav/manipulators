# Detection System Diagrams

Mermaid diagrams for the object detection node, detection pipeline, camera calibration, and coordinate transforms.

---

## 1. Object Detection Node Architecture

Internal structure of the detection node with pluggable detector system.

```mermaid
graph TB
    subgraph ROS2 Subscriptions
        C1[/color/image_raw]
        C2[/aligned_depth]
        C3[/camera_info]
    end

    subgraph object_detection_node
        SYNC[message_filters<br>ApproximateTimeSynchronizer]
        DET[Pluggable Detector<br>ColorDetector / YOLODetector / ...]
        PROJ[3D Projection<br>pixel → camera → robot]
        CAL[Camera Calibration<br>intrinsics + extrinsics]
        VIZ[Visualization<br>annotated image]
    end

    subgraph ROS2 Publishers
        P1[/detected_object_point<br>PointStamped]
        P2[/detection_image<br>Image]
    end

    C1 --> SYNC
    C2 --> SYNC
    C3 --> SYNC

    SYNC --> DET
    DET -- "Detection(u, v, label)" --> PROJ
    C2 --> PROJ
    CAL --> PROJ
    PROJ --> P1

    DET --> VIZ
    VIZ --> P2

    style DET fill:#fff9c4
    style PROJ fill:#e1f5fe
```

---

## 2. Detection Pipeline — Single Frame

What happens when a synchronized color + depth frame arrives.

```mermaid
flowchart TD
    A[Synchronized callback:<br>color_msg + depth_msg] --> B[Convert to OpenCV<br>cv_bridge]
    B --> C[Run detector.detect<br>color_image]

    subgraph ColorDetector.detect
        C --> D1[Apply crop ROI]
        D1 --> D2[BGR range filter]
        D2 --> D3[Convert to HSV]
        D3 --> D4[HSV range filter]
        D4 --> D5[AND masks]
        D5 --> D6[Find contours]
        D6 --> D7[Filter by min_area]
        D7 --> D8[Find largest contour]
        D8 --> D9[Return centroid as Detection]
    end

    D9 --> E{Detection found?}
    E -- Yes --> F[Read depth at u, v]
    F --> G{Depth valid?<br>within range}
    G -- Yes --> H[Project to 3D camera frame<br>X, Y, Z from u, v, depth]
    H --> I[Transform to robot frame<br>using extrinsic R, t]
    I --> J[Publish PointStamped]

    E -- No --> K[Skip frame]
    G -- No --> K

    J --> L[Draw bounding box + label]
    K --> L
    L --> M[Publish detection_image]
```

---

## 3. Camera Calibration Workflow

Steps to calibrate camera intrinsics and extrinsics.

```mermaid
flowchart TD
    subgraph "1. Capture Images"
        A1[Start realsense2_camera]
        A2[Run capture_images.py]
        A3[Move chessboard to<br>various poses]
        A4[Press SPACE to capture<br>20+ images]
        A1 --> A2 --> A3 --> A4
    end

    subgraph "2. Intrinsic Calibration"
        B1[Run calibrate_intrinsics.py]
        B2[Detect chessboard corners<br>in all images]
        B3[OpenCV calibrateCamera]
        B4[Output: camera_matrix<br>dist_coeffs]
        B1 --> B2 --> B3 --> B4
    end

    subgraph "3. Extrinsic Calibration"
        C1[Place chessboard at<br>known robot pose]
        C2[Run calibrate_extrinsics.py]
        C3[Detect corners + solvePnP]
        C4[Compute camera → chessboard]
        C5[Combine with known<br>chessboard → robot]
        C6[Output: rvec, tvec<br>camera → robot]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end

    A4 --> B1
    B4 --> C2
    B4 --> CAM[config/camera.yaml]
    C6 --> CAM

    style CAM fill:#c8e6c9
```

---

## 4. Pluggable Detector System

Class diagram showing how to add new detectors.

```mermaid
classDiagram
    class DetectorBase {
        <<Protocol>>
        +detect(color_image: ndarray) List~Detection~
    }

    class Detection {
        +u: int
        +v: int
        +label: str
        +confidence: float
        +bbox: Optional~Tuple~
        +mask: Optional~ndarray~
    }

    class ColorDetector {
        -hsv_low: ndarray
        -hsv_high: ndarray
        -bgr_low: ndarray
        -bgr_high: ndarray
        -crop: Tuple
        -min_area: int
        +detect(color_image) List~Detection~
    }

    class YOLODetector {
        <<future>>
        -model: YOLO
        -confidence_threshold: float
        +detect(color_image) List~Detection~
    }

    DetectorBase <|.. ColorDetector
    DetectorBase <|.. YOLODetector
    ColorDetector --> Detection
    YOLODetector --> Detection
```

---

## 5. Coordinate Frame Transforms

How 2D pixel coordinates become 3D robot-frame positions.

```mermaid
flowchart LR
    subgraph "Image Plane"
        PX["(u, v)<br>pixel coords"]
    end

    subgraph "Depth Lookup"
        D["depth[v, u]<br>meters"]
    end

    subgraph "Camera Frame"
        PC["(X_c, Y_c, Z_c)<br>camera 3D"]
    end

    subgraph "Robot Frame"
        PR["(X_r, Y_r, Z_r)<br>robot 3D"]
    end

    PX --> UNPROJ["Unproject:<br>X = (u - cx) * Z / fx<br>Y = (v - cy) * Z / fy<br>Z = depth"]
    D --> UNPROJ
    UNPROJ --> PC

    PC --> TRANS["Transform:<br>P_robot = R * P_cam + t<br><br>R, t from extrinsics"]
    TRANS --> PR

    style UNPROJ fill:#e1f5fe
    style TRANS fill:#fff9c4
```

---

## 6. Detection Data Flow

How detection integrates with the rest of the system.

```mermaid
flowchart TB
    subgraph Camera Driver
        RS[realsense2_camera]
    end

    subgraph Detection Node
        OD[object_detection_node]
    end

    subgraph Consumers
        PP[pick_place_policy]
        VIZ[RViz / rqt_image_view]
    end

    RS -- "/color/image_raw<br>sensor_msgs/Image" --> OD
    RS -- "/aligned_depth_to_color/image_raw<br>sensor_msgs/Image" --> OD
    RS -- "/camera_info<br>sensor_msgs/CameraInfo" --> OD

    OD -- "/detected_object_point<br>geometry_msgs/PointStamped" --> PP
    OD -- "/detection_image<br>sensor_msgs/Image" --> VIZ

    style OD fill:#e3f2fd
    style PP fill:#fff9c4
```
