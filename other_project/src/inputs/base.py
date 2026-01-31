"""
Base class for input plugins.

All input sources (keyboard, OSC, MIDI, gamepad) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from threading import Thread, Event

from ..core.bus import MessageBus


class InputPlugin(ABC):
    """
    Base class for input source plugins.

    Subclasses must implement:
    - name: Unique identifier for this input source
    - setup(): Initialize the input (open ports, etc.)
    - loop(): Process input and publish to bus
    - teardown(): Cleanup resources
    """

    def __init__(self, bus: MessageBus, config: dict):
        """
        Initialize input plugin.

        Args:
            bus: Message bus for publishing PoseDelta messages
            config: Plugin-specific configuration
        """
        self.bus = bus
        self.config = config

        self._thread: Optional[Thread] = None
        self._running = Event()
        self._enabled = Event()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this input source."""
        pass

    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the input source.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def loop(self) -> None:
        """
        Process input and publish messages.

        Called repeatedly while enabled. Should check _enabled flag
        and only publish when enabled.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup resources."""
        pass

    def start(self) -> None:
        """Start the input plugin thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._running.set()
        self._thread = Thread(
            target=self._run,
            name=f"Input-{self.name}",
            daemon=True
        )
        self._thread.start()
        print(f"[Input:{self.name}] Started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the input plugin."""
        self._running.clear()
        self._enabled.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        print(f"[Input:{self.name}] Stopped")

    def enable(self) -> None:
        """Enable input processing (publish messages)."""
        self._enabled.set()
        print(f"[Input:{self.name}] Enabled")

    def disable(self) -> None:
        """Disable input processing (stop publishing)."""
        self._enabled.clear()
        print(f"[Input:{self.name}] Disabled")

    def is_enabled(self) -> bool:
        """Check if input is enabled."""
        return self._enabled.is_set()

    def _run(self) -> None:
        """Internal thread entry point."""
        if not self.setup():
            print(f"[Input:{self.name}] Setup failed")
            self._running.clear()
            return

        try:
            while self._running.is_set():
                self.loop()
        finally:
            self.teardown()
