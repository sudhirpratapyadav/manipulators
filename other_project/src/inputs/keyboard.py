"""
Keyboard input plugin using pynput.

Maps WASD/QE keys to position, IJKL/UO to rotation.
"""

import time
from typing import Dict, Tuple

from ..core.bus import MessageBus, Topics
from ..core.messages import PoseDelta
from .base import InputPlugin


class KeyboardInput(InputPlugin):
    """
    Keyboard input using pynput.

    Key mappings:
    - W/S: +/- X position
    - A/D: +/- Y position
    - Q/E: +/- Z position
    - I/K: +/- Rx rotation
    - J/L: +/- Ry rotation
    - U/O: +/- Rz rotation
    - H: Reset to home (special signal)
    """

    def __init__(self, bus: MessageBus, config: dict):
        super().__init__(bus, config)

        # Step sizes from config
        self._pos_step = config.get("position_step_m", 0.05)
        self._rot_step = config.get("rotation_step_rad", 0.1)

        # Key mappings: key -> (axis_index, sign)
        # Axes 0-2 are position (x, y, z), 3-5 are rotation (rx, ry, rz)
        self._key_map: Dict[str, Tuple[int, int]] = {
            'w': (0, +1), 's': (0, -1),  # X
            'a': (1, +1), 'd': (1, -1),  # Y
            'q': (2, +1), 'e': (2, -1),  # Z
            'i': (3, +1), 'k': (3, -1),  # Rx
            'j': (4, +1), 'l': (4, -1),  # Ry
            'u': (5, +1), 'o': (5, -1),  # Rz
        }

        self._listener = None
        self._pynput_available = False

    @property
    def name(self) -> str:
        return "keyboard"

    def setup(self) -> bool:
        """Initialize pynput listener."""
        try:
            from pynput import keyboard

            def on_press(key):
                if not self._running.is_set():
                    return False  # Stop listener

                if not self._enabled.is_set():
                    return True  # Ignore but keep listening

                try:
                    char = key.char.lower() if hasattr(key, 'char') and key.char else None

                    if char == 'h':
                        # Special: reset to home
                        # Publish a large negative delta to signal reset (handled by IKActor)
                        # Actually, we should handle this differently - for now just print
                        print("[Keyboard] Home key pressed")
                        return True

                    if char in self._key_map:
                        axis, sign = self._key_map[char]
                        self._publish_delta(axis, sign)

                except Exception:
                    pass

                return True

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            self._pynput_available = True

            print("[Keyboard] Controls:")
            print("  Position: W/S (X), A/D (Y), Q/E (Z)")
            print("  Rotation: I/K (Rx), J/L (Ry), U/O (Rz)")

            return True

        except ImportError:
            print("[Keyboard] pynput not available - keyboard input disabled")
            self._pynput_available = False
            return True  # Still return True to not block startup

    def loop(self) -> None:
        """Just sleep - pynput handles events in its own thread."""
        time.sleep(0.05)

    def _publish_delta(self, axis: int, sign: int) -> None:
        """Publish a pose delta for the given axis."""
        delta_pos = [0.0, 0.0, 0.0]
        delta_rot = [0.0, 0.0, 0.0]

        if axis < 3:
            delta_pos[axis] = sign * self._pos_step
        else:
            delta_rot[axis - 3] = sign * self._rot_step

        msg = PoseDelta(
            delta_position=tuple(delta_pos),
            delta_orientation=tuple(delta_rot),
            source="keyboard",
        )
        self.bus.publish(Topics.INPUT_DELTA, msg)

    def teardown(self) -> None:
        """Stop the listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None
