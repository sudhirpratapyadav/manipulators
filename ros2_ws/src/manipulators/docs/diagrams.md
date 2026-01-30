# Manipulators — Diagrams

Mermaid diagrams for the `manipulators` package architecture and behavior.

---

## 1. System Architecture

High-level view of nodes, topics, services, and the hardware layer.

```mermaid
graph TB
    subgraph Input Nodes
        KT[keyboard_teleop]
        PN[policy_node<br><i>future</i>]
        CN[camera_node<br><i>future</i>]
    end

    subgraph control_node
        CB[ROS2 Callbacks<br><i>main thread</i>]
        CL[Control Loop<br><i>dedicated thread @ 400Hz</i>]
        CTRL[DiffIKController]
        PM[RobotModel<br><i>Pinocchio</i>]
        HW[KinovaHardware<br><i>Kortex API</i>]
    end

    subgraph Kinova Gen3
        TCP[TCP :10000<br><i>high-level cmds</i>]
        UDP[UDP :10001<br><i>real-time torques</i>]
        GRIP[Robotiq 2F-85<br><i>via interconnect</i>]
    end

    KT -- /target_pose<br>PoseStamped --> CB
    KT -- /gripper_command<br>Float64 --> CB
    PN -. /target_pose .-> CB
    CN -. /detections .-> PN

    CB -- "lock-protected<br>shared state" --> CL
    CL --> CTRL
    CTRL --> PM
    CL --> HW

    HW -- "torques + gripper" --> UDP
    HW -- "home, mode switch" --> TCP
    UDP -- feedback --> HW
    HW --> GRIP

    CL -- /joint_states --> EXT[ ]
    CL -- /ee_pose --> EXT
    CB -- /e_stop --> CL

    style EXT fill:none,stroke:none
    style CN stroke-dasharray: 5 5
    style PN stroke-dasharray: 5 5
```

---

## 2. Control Node Startup Sequence

Step-by-step initialization from launch to control loop running.

```mermaid
sequenceDiagram
    participant L as Launch System
    participant N as ControlNode
    participant HW as KinovaHardware
    participant K as Kinova Robot
    participant P as Pinocchio
    participant C as DiffIKController

    L->>N: Create node, load parameters
    N->>HW: connect(ip, user, pass)
    HW->>K: TCP connect :10000
    HW->>K: UDP connect :10001
    HW-->>N: Connected

    N->>HW: clear_faults()
    HW->>K: ClearFaults()
    N->>HW: wait_until_ready()
    HW->>K: GetArmState() [poll]
    HW-->>N: Ready

    N->>HW: go_to_joints(home_deg)
    HW->>K: ExecuteAction(JointMove)
    K-->>HW: ACTION_END
    HW-->>N: Home reached

    N->>P: RobotModel(urdf_path)
    P-->>N: 7-DOF reduced model

    N->>HW: read_state()
    HW->>K: RefreshFeedback()
    K-->>HW: positions, velocities, torques
    HW-->>N: RobotState
    N->>P: fk(q)
    P-->>N: EE pose (initial target)

    N->>C: DiffIKController(model, gains)

    N->>HW: set_servoing_mode(LOW_LEVEL)
    HW->>K: SetServoingMode
    Note over N: sleep 0.5s
    N->>HW: set_torque_mode(True)
    HW->>K: SetControlMode(TORQUE) x7

    N->>N: Start control loop thread
    Note over N: Running @ 400Hz
```

---

## 3. Control Node Shutdown Sequence

Clean shutdown from SIGINT to disconnect.

```mermaid
sequenceDiagram
    participant U as User / SIGINT
    participant N as ControlNode
    participant HW as KinovaHardware
    participant K as Kinova Robot

    U->>N: Ctrl+C / KeyboardInterrupt

    N->>N: _running = False
    N->>N: control_thread.join()
    Note over N: Loop stopped

    N->>HW: set_torque_mode(False)
    HW->>K: SetControlMode(POSITION) x7
    Note over N: sleep 0.5s

    N->>HW: set_servoing_mode(HIGH_LEVEL)
    HW->>K: SetServoingMode(SINGLE_LEVEL)
    Note over N: sleep 1.0s

    N->>HW: clear_faults()
    N->>HW: wait_until_ready()
    N->>HW: go_to_joints(home_deg)
    HW->>K: ExecuteAction(JointMove)
    K-->>HW: ACTION_END

    N->>HW: disconnect()
    HW->>K: CloseSession (UDP)
    HW->>K: CloseSession (TCP)
    HW->>K: Disconnect transports
    N->>N: destroy_node()
```

---

## 4. Control Loop — Single Cycle

What happens every ~2.5ms (400Hz) inside the control thread.

```mermaid
flowchart TD
    A[Start cycle<br>t_start = perf_counter] --> B[Convert state to radians<br>q, dq = from positions_deg]
    B --> C[Lock: read target_pos,<br>target_quat, gripper]
    C --> D[controller.compute]

    subgraph DiffIKController.compute
        D --> D1[FK: current EE pose]
        D1 --> D2[Pose error: 6D<br>position + orientation]
        D2 --> D3["Desired twist = Kp_task * error"]
        D3 --> D4["Jacobian J(q) — 6x7"]
        D4 --> D5["dq_desired = J_pinv * twist<br>(damped pseudoinverse)"]
        D5 --> D6[Clamp dq_desired<br>by max_joint_velocity]
        D6 --> D7["q_desired = q + dq_desired * dt"]
        D7 --> D8["tau = Kp_joint*(q_des - q)<br>+ Kd_joint*(dq_des - dq)<br>+ gravity(q)"]
        D8 --> D9[Clamp tau by max_torque]
    end

    D9 --> E["hw.send_torques(tau, pos_deg, gripper)<br>UDP Refresh — returns new state"]
    E --> F[Publish /joint_states, /ee_pose]
    F --> G[Sleep remaining time<br>dt - elapsed]
    G --> A
```

---

## 5. E-Stop Sequence

Emergency stop triggered via ROS2 service.

```mermaid
sequenceDiagram
    participant U as User / External Node
    participant CB as Callback (main thread)
    participant CL as Control Loop (thread)
    participant HW as KinovaHardware
    participant K as Kinova Robot

    U->>CB: ros2 service call /e_stop
    CB->>CL: _running = False
    CB->>HW: set_torque_mode(False)
    HW->>K: SetControlMode(POSITION) x7
    CB->>HW: stop()
    HW->>K: Stop()
    CB-->>U: success=True, "E-stop executed"
    Note over CL: Loop exits on next iteration
```

---

## 6. Threading Model

How the main thread and control thread interact.

```mermaid
graph LR
    subgraph "Main Thread (rclpy.spin)"
        S1["/target_pose callback"]
        S2["/gripper_command callback"]
        S3["/e_stop service handler"]
    end

    subgraph "Shared State (threading.Lock)"
        SS["_target_pos (3,)<br>_target_quat (4,)<br>_gripper_target<br>_running"]
    end

    subgraph "Control Thread (dedicated)"
        CL["400Hz loop:<br>read state → compute → send torques → publish"]
    end

    S1 -- "write under lock" --> SS
    S2 -- "write under lock" --> SS
    S3 -- "set _running=False" --> SS
    SS -- "read under lock<br>(copy out)" --> CL
```

---

## 7. Module Dependency Graph

Import relationships between package modules.

```mermaid
graph TD
    CN[control_node.py] --> HW[hardware.py]
    CN --> RM[robot_model.py]
    CN --> DI[diff_ik_controller.py]
    CN --> UT[utility.py]

    DI --> RM
    DI --> UT

    KT[keyboard_teleop.py] --> UT

    RM --> PIN[pinocchio]
    HW --> KORTEX[kortex_api]
    UT --> PIN

    style PIN fill:#e1f5fe
    style KORTEX fill:#fce4ec
```

---

## 8. Data Flow — Coordinate Conversions

How joint data is transformed between Kinova, Pinocchio, and ROS2.

```mermaid
flowchart LR
    subgraph Kinova Hardware
        KD["positions_deg<br>0-360 range"]
        KV["velocities_deg<br>deg/s"]
        KT["torques<br>Nm"]
    end

    subgraph utility.py
        CONV["kinova_degrees_to_radians()<br>0-360 → signed → radians"]
    end

    subgraph Pinocchio / Controller
        Q["q (rad)<br>-pi to pi"]
        DQ["dq (rad/s)"]
        TAU["tau (Nm)"]
    end

    subgraph ROS2 Topics
        JS["/joint_states<br>position: rad<br>velocity: rad/s<br>effort: Nm"]
        EP["/ee_pose<br>PoseStamped"]
    end

    KD --> CONV --> Q
    KV -- "np.deg2rad" --> DQ
    Q --> TAU
    DQ --> TAU
    TAU -- "send_torques()" --> KD
    Q --> JS
    DQ --> JS
    KT --> JS
    Q -- "fk(q)" --> EP
```

---

## 9. Kinova Hardware — Connection Architecture

TCP and UDP dual-connection model.

```mermaid
graph TB
    subgraph KinovaHardware
        direction TB
        TCP_S["TCP Session<br>:10000"]
        UDP_S["UDP Session<br>:10001"]

        BC[BaseClient]
        AC[ActuatorConfigClient]
        BCC[BaseCyclicClient]

        TCP_S --> BC
        TCP_S --> AC
        UDP_S --> BCC
    end

    subgraph "TCP — High Level (non-real-time)"
        BC --> HL1[ClearFaults]
        BC --> HL2[GetArmState]
        BC --> HL3[SetServoingMode]
        BC --> HL4[ExecuteAction<br>JointMove]
        BC --> HL5[SendGripperCommand]
        BC --> HL6[Stop]
        AC --> HL7[SetControlMode<br>TORQUE / POSITION]
    end

    subgraph "UDP — Low Level (real-time @ 1kHz)"
        BCC --> LL1["RefreshFeedback()<br>read-only"]
        BCC --> LL2["Refresh(cmd)<br>send torques +<br>read feedback"]
    end

    style TCP_S fill:#e8f5e9
    style UDP_S fill:#fff3e0
```
