# Kinova Gen3 Controller v2 - Message-Passing Architecture

A complete rewrite of the controller using a message-passing actor model.

## Quick Start

```bash
# Run with default config
python -m src.main

# Run with custom config
python -m src.main --config config/default.yaml

# Run with different robot IP
python -m src.main --ip 192.168.1.20
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MESSAGE BUS                                  │
│  Topics: /robot/state, /control/torque, /input/delta, etc.         │
└───────────┬──────────────────┬──────────────────┬───────────────────┘
            │                  │                  │
   ┌────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
   │  HardwareActor  │ │ ControlActor  │ │    IKActor      │
   │   (1kHz UDP)    │ │  (1kHz PD)    │ │   (400Hz IK)    │
   └─────────────────┘ └───────────────┘ └─────────────────┘
            │                  │                  │
   ┌────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
   │  SafetyActor    │ │  StateActor   │ │  KeyboardInput  │
   │ (1kHz monitor)  │ │  (50Hz agg)   │ │   (keyboard)    │
   └─────────────────┘ └───────────────┘ └─────────────────┘
```

## Key Concepts

### 1. Message Bus
All communication happens through typed messages on topics:

```python
from core.bus import MessageBus, Topics
from core.messages import RobotState, TorqueCommand

bus = MessageBus()

# Publish
bus.publish(Topics.ROBOT_STATE, RobotState(
    joint_positions=(0.1, 0.2, ...),
    joint_velocities=(0.0, 0.0, ...),
))

# Subscribe (callback)
def on_state(msg: RobotState):
    print(msg.joint_positions)
bus.subscribe_callback(Topics.ROBOT_STATE, on_state)

# Subscribe (queue)
queue = bus.subscribe_queue(Topics.ROBOT_STATE)
msg = queue.get()
```

### 2. Actors
Independent processing units with their own loop:

```python
from core.actor import TimedActor

class MyActor(TimedActor):
    def setup(self):
        self.queue = self.bus.subscribe_queue("/my/topic")

    def loop(self):
        msg = self.queue.get_nowait()
        result = process(msg)
        self.bus.publish("/my/output", result)

    def teardown(self):
        pass
```

### 3. Input Plugins
Extensible input sources:

```python
from inputs.base import InputPlugin

class GamepadInput(InputPlugin):
    @property
    def name(self):
        return "gamepad"

    def setup(self):
        self.gamepad = connect_gamepad()
        return True

    def loop(self):
        if self.is_enabled():
            axes = self.gamepad.read()
            self.bus.publish(Topics.INPUT_DELTA, PoseDelta(...))
```

## Directory Structure

```
src/
├── core/                   # Framework essentials
│   ├── messages.py         # Typed message definitions
│   ├── bus.py              # Message bus implementation
│   ├── actor.py            # Actor base classes
│   └── config.py           # Configuration management
│
├── actors/                 # Processing units
│   ├── hardware_actor.py   # 1kHz UDP I/O
│   ├── control_actor.py    # Gravity + PD control
│   ├── ik_actor.py         # 400Hz differential IK
│   ├── safety_actor.py     # Velocity/torque limits
│   └── state_actor.py      # State aggregation for UI
│
├── inputs/                 # Input source plugins
│   ├── base.py             # InputPlugin interface
│   └── keyboard.py         # pynput keyboard
│
├── robot/                  # Robot-specific code
│   ├── kinova.py           # Kortex API wrapper
│   └── model.py            # Pinocchio kinematics
│
├── assets/
│   └── kinova/             # URDF and meshes
│       ├── gen3_2f85.urdf
│       └── meshes/
│
├── gui/                    # Visualization
│   └── server.py           # Viser web interface
│
└── main.py                 # Orchestrator
```

## Message Flow

### Normal Operation (Diff-IK Mode)

```
Keyboard Press
    │
    ▼
KeyboardInput.loop()
    │ publishes PoseDelta to /input/delta
    ▼
IKActor.loop() (400Hz)
    │ accumulates deltas
    │ computes IK
    │ publishes DesiredJoints to /control/desired
    ▼
ControlActor.loop() (1kHz)
    │ reads desired joints
    │ computes gravity + PD
    │ publishes TorqueCommand to /control/torque
    ▼
HardwareActor.loop() (1kHz)
    │ reads TorqueCommand
    │ sends to robot
    │ reads feedback
    │ publishes RobotState to /robot/state
    ▼
StateActor.loop() (50Hz)
    │ aggregates state
    │ publishes UIState to /ui/state
    ▼
GUIServer.update() (50Hz)
    │ updates displays
    └─────────────────────
```

## Configuration

Edit `config/default.yaml`:

```yaml
kinova:
  ip: "192.168.1.10"

control:
  rates:
    hardware_hz: 1000
    ik_hz: 400

  gains:
    kp: [120, 120, 120, 120, 80, 80, 80]
    kd: [12, 12, 12, 12, 8, 8, 2]

inputs:
  keyboard:
    position_step_m: 0.05
    rotation_step_rad: 0.1
```

## Adding New Input Sources

1. Create `inputs/my_input.py`:

```python
from .base import InputPlugin

class MyInput(InputPlugin):
    @property
    def name(self):
        return "my_input"

    def setup(self):
        # Initialize your input device
        return True

    def loop(self):
        if self.is_enabled():
            # Read input and publish pose deltas
            self.bus.publish(Topics.INPUT_DELTA, PoseDelta(...))
```

2. Add to `main.py`:
```python
from .inputs.my_input import MyInput

self.my_input = MyInput(self.bus, config)
self.my_input.start()
```

## Performance

| Component | Overhead |
|-----------|----------|
| Message publish | ~1-2 μs |
| Queue get | ~0.5 μs |
| Full control loop | ~400-600 μs |

The message bus adds negligible overhead (<1%) compared to UDP I/O.

## Comparison to v1

| Aspect | v1 (SharedState) | v2 (Messages) |
|--------|------------------|---------------|
| Communication | Shared mutable state | Typed messages |
| Coupling | Implicit (all touch state) | Explicit (subscriptions) |
| Testing | Hard (need full system) | Easy (mock bus) |
| Adding inputs | Modify core files | Add plugin |
| Configuration | Edit Python | Edit YAML |
| Debug | Print statements | Message tracing |
