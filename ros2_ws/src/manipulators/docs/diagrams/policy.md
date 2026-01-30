# Pick-Place Policy Diagrams

Mermaid diagrams for the reactive pick-and-place policy, state machine, and motion control.

---

## 1. Policy Node Architecture

How the pick-place policy integrates with the system.

```mermaid
graph TB
    subgraph Inputs
        OD[object_detection_node]
        CN[control_node]
    end

    subgraph pick_place_policy
        FSM[State Machine<br>12 states]
        VEL[Velocity Limiter<br>smooth target updates]
        GRIP[Gripper Controller]
    end

    subgraph Outputs
        TGT[/target_pose]
        GRP[/gripper_command]
    end

    subgraph Services
        START[/pick_place/start]
        ABORT[/pick_place/abort]
    end

    OD -- "/detected_object_point<br>PointStamped" --> FSM
    CN -- "/ee_pose<br>PoseStamped" --> FSM

    FSM --> VEL
    FSM --> GRIP
    VEL --> TGT
    GRIP --> GRP

    START --> FSM
    ABORT --> FSM

    style FSM fill:#fff9c4
    style VEL fill:#e1f5fe
```

---

## 2. State Machine — Full Diagram

Complete state machine with all transitions.

```mermaid
stateDiagram-v2
    [*] --> INIT: Node startup

    INIT --> IDLE: Position reached

    IDLE --> APPROACH: /pick_place/start<br>(fresh detection required)

    APPROACH --> DESCEND: Pre-grasp reached
    DESCEND --> GRASP: Grasp position reached

    GRASP --> LIFT: Settle time elapsed

    LIFT --> MID_TRANSPORT: Lift height reached
    MID_TRANSPORT --> PRE_PLACE: Mid-transport reached

    PRE_PLACE --> PLACE_DESCEND: Pre-place reached
    PLACE_DESCEND --> RELEASE: Place position reached

    RELEASE --> PLACE_ASCEND: Settle time elapsed

    PLACE_ASCEND --> RETURN_IDLE: Pre-place reached
    RETURN_IDLE --> IDLE: Idle position reached

    note right of IDLE
        Waiting for service trigger.
        Object detection cleared.
    end note

    note right of APPROACH
        Tracks object position.
        Target updates reactively.
    end note

    note right of GRASP
        Gripper closes.
        Object position latched.
    end note

    note right of RELEASE
        Gripper opens.
        Hold position.
    end note
```

---

## 3. State Machine — Linear Flow

Simplified linear view of the pick-place cycle.

```mermaid
flowchart LR
    INIT --> IDLE
    IDLE --> APPROACH
    APPROACH --> DESCEND
    DESCEND --> GRASP
    GRASP --> LIFT
    LIFT --> MID_TRANSPORT
    MID_TRANSPORT --> PRE_PLACE
    PRE_PLACE --> PLACE_DESCEND
    PLACE_DESCEND --> RELEASE
    RELEASE --> PLACE_ASCEND
    PLACE_ASCEND --> RETURN_IDLE
    RETURN_IDLE --> IDLE

    style IDLE fill:#c8e6c9
    style GRASP fill:#ffccbc
    style RELEASE fill:#ffccbc
    style APPROACH fill:#e1f5fe
    style DESCEND fill:#e1f5fe
```

---

## 4. Velocity-Limited Target Updates

How the policy creates smooth motion by rate-limiting target changes.

```mermaid
flowchart TD
    A[Compute desired_target<br>from current state] --> B[Compute direction<br>desired - current]
    B --> C[Compute distance<br>||direction||]
    C --> D{distance > max_step?}

    D -- Yes --> E["Move toward target:<br>current += (dir/dist) * max_step"]
    D -- No --> F["Snap to target:<br>current = desired"]

    E --> G[Publish /target_pose]
    F --> G

    G --> H[Control node tracks<br>smoothly moving target]

    subgraph "Result: Smooth Motion"
        H --> I[No sudden jumps]
        H --> J[Natural acceleration/<br>deceleration]
        H --> K[Safe behavior for<br>distant objects]
    end

    style D fill:#fff9c4
    style E fill:#e1f5fe
    style F fill:#c8e6c9
```

---

## 5. State Transition Conditions

What triggers each state transition.

```mermaid
flowchart TD
    subgraph "Position-Based Transitions"
        T1["INIT → IDLE<br>||ee - idle_pos|| < threshold"]
        T2["APPROACH → DESCEND<br>||ee - pre_grasp|| < threshold"]
        T3["DESCEND → GRASP<br>||ee - grasp_pos|| < threshold"]
        T4["LIFT → MID_TRANSPORT<br>||ee - lift_pos|| < threshold"]
        T5["MID_TRANSPORT → PRE_PLACE<br>||ee - mid_transport|| < threshold"]
        T6["PRE_PLACE → PLACE_DESCEND<br>||ee - pre_place|| < threshold"]
        T7["PLACE_DESCEND → RELEASE<br>||ee - place_pos|| < threshold"]
        T8["PLACE_ASCEND → RETURN_IDLE<br>||ee - pre_place|| < threshold"]
        T9["RETURN_IDLE → IDLE<br>||ee - idle_pos|| < threshold"]
    end

    subgraph "Time-Based Transitions"
        T10["GRASP → LIFT<br>time_in_state > grasp_settle_time"]
        T11["RELEASE → PLACE_ASCEND<br>time_in_state > release_settle_time"]
    end

    subgraph "Service-Triggered Transitions"
        T12["IDLE → APPROACH<br>/pick_place/start<br>+ fresh object detection"]
        T13["ANY → RETURN_IDLE<br>/pick_place/abort"]
    end

    style T10 fill:#ffccbc
    style T11 fill:#ffccbc
    style T12 fill:#c8e6c9
    style T13 fill:#ffcdd2
```

---

## 6. Waypoint Positions

Spatial layout of all waypoints in the pick-place cycle.

```mermaid
flowchart TB
    subgraph "Z-axis (height)"
        direction TB
        Z1[idle_position.z]
        Z2[pre_grasp_height<br>above object]
        Z3[grasp_height<br>at object]
        Z4[lift_height<br>above object]
        Z5[mid_transport.z]
        Z6[pre_place.z]
        Z7[place.z]
    end

    subgraph "Pick-Place Path"
        IDLE[IDLE<br>idle_position] --> PRE[PRE_GRASP<br>above object]
        PRE --> GR[GRASP<br>at object]
        GR --> LI[LIFT<br>above object]
        LI --> MID[MID_TRANSPORT<br>waypoint]
        MID --> PREP[PRE_PLACE<br>above place]
        PREP --> PL[PLACE<br>at place]
        PL --> PREP2[PRE_PLACE<br>ascend]
        PREP2 --> IDLE2[IDLE<br>return]
    end

    style GR fill:#ffccbc
    style PL fill:#c8e6c9
    style MID fill:#e1f5fe
```

---

## 7. Object Tracking Behavior

When and how object position is used.

```mermaid
flowchart TD
    subgraph "Object Tracking Active"
        A1[APPROACH state]
        A2[DESCEND state]
        A1 --> T1[Target = object_pos + offset]
        A2 --> T1
        T1 --> U1[Updates every cycle<br>as object moves]
    end

    subgraph "Object Position Latched"
        B1[GRASP state entry]
        B1 --> L1[grasp_pos = last_valid_object_pos]
        L1 --> L2[Used for LIFT calculation]
    end

    subgraph "Object Position Ignored"
        C1[LIFT]
        C2[MID_TRANSPORT]
        C3[PRE_PLACE]
        C4[PLACE_DESCEND]
        C5[RELEASE]
        C6[PLACE_ASCEND]
        C7[RETURN_IDLE]
        C1 --> I1[Fixed waypoints only]
        C2 --> I1
        C3 --> I1
        C4 --> I1
        C5 --> I1
        C6 --> I1
        C7 --> I1
    end

    subgraph "Detection Cleared"
        D1[IDLE state entry]
        D1 --> CLR[object_pos = None<br>detection_time = None]
        CLR --> REQ[Fresh detection required<br>for next cycle]
    end

    style A1 fill:#e1f5fe
    style A2 fill:#e1f5fe
    style B1 fill:#ffccbc
    style D1 fill:#c8e6c9
```

---

## 8. Service Interface

How external nodes trigger pick-place operations.

```mermaid
sequenceDiagram
    participant U as User / Orchestrator
    participant P as pick_place_policy
    participant D as object_detection_node
    participant C as control_node

    Note over P: State: IDLE

    U->>P: /pick_place/start
    P->>P: Check state == IDLE
    P->>P: Check detection fresh?
    alt No fresh detection
        P-->>U: success=False<br>"no fresh object detection"
    else Detection available
        P->>P: Transition to APPROACH
        P-->>U: success=True<br>"Pick-place cycle started"
    end

    loop Every 20ms (50Hz)
        D->>P: /detected_object_point
        C->>P: /ee_pose
        P->>P: Update state machine
        P->>C: /target_pose
        P->>C: /gripper_command
    end

    Note over P: State: IDLE (cycle complete)

    U->>P: /pick_place/abort
    alt Not in IDLE
        P->>P: Open gripper
        P->>P: Transition to RETURN_IDLE
        P-->>U: success=True<br>"Aborting, returning to idle"
    else Already in IDLE
        P-->>U: success=False<br>"Nothing to abort"
    end
```

---

## 9. Configuration Parameters

Visual guide to configurable waypoints and thresholds.

```mermaid
flowchart TB
    subgraph "Position Parameters"
        IP[idle_position<br>x, y, z]
        MTP[mid_transport_position<br>x, y, z]
        PPP[pre_place_position<br>x, y, z]
        PP[place_position<br>x, y, z]
    end

    subgraph "Height Parameters"
        PGH[pre_grasp_height<br>10cm default]
        GH[grasp_height<br>2.5cm default]
        LH[lift_height<br>15cm default]
    end

    subgraph "Motion Parameters"
        MLV[max_linear_velocity<br>0.25 m/s]
        MAV[max_angular_velocity<br>1.0 rad/s]
        PT[position_threshold<br>1cm]
        OT[orientation_threshold<br>0.05 rad]
    end

    subgraph "Timing Parameters"
        GST[grasp_settle_time<br>0.8s]
        RST[release_settle_time<br>0.5s]
        DTO[detection_timeout<br>1.0s]
    end

    subgraph "Orientation"
        GO[grasp_orientation<br>quaternion xyzw<br>top-down default]
    end

    style IP fill:#c8e6c9
    style PP fill:#c8e6c9
    style MLV fill:#e1f5fe
    style GST fill:#ffccbc
```

---

## 10. Full System Integration

Complete data flow from detection to robot motion.

```mermaid
flowchart TB
    subgraph Camera
        RS[RealSense D435]
    end

    subgraph Detection
        OD[object_detection_node]
    end

    subgraph Policy
        PP[pick_place_policy<br>FSM + Velocity Limiter]
    end

    subgraph Control
        CN[control_node<br>Diff-IK @ 400Hz]
    end

    subgraph Robot
        K[Kinova Gen3<br>+ Robotiq 2F-85]
    end

    RS -- "RGB + Depth<br>30Hz" --> OD
    OD -- "/detected_object_point<br>30Hz" --> PP

    CN -- "/ee_pose<br>400Hz" --> PP

    PP -- "/target_pose<br>50Hz" --> CN
    PP -- "/gripper_command<br>50Hz" --> CN

    CN -- "Joint torques<br>UDP 400Hz" --> K
    K -- "Joint state<br>UDP 400Hz" --> CN

    style PP fill:#fff9c4
    style CN fill:#e3f2fd
    style OD fill:#e3f2fd
```
