"""
Base Actor class for message-passing architecture.

Actors are independent processing units that:
- Have their own thread/loop
- Communicate only via the message bus
- Don't share mutable state with other actors
"""

from abc import ABC, abstractmethod
from threading import Thread, Event
from typing import Optional
import time

from .bus import MessageBus


class Actor(ABC):
    """
    Base class for all actors.

    Subclasses must implement:
    - setup(): Initialize resources
    - loop(): Main processing (called repeatedly)
    - teardown(): Cleanup resources
    """

    def __init__(self, name: str, bus: MessageBus, config: dict):
        """
        Initialize actor.

        Args:
            name: Actor name (for logging)
            bus: Message bus for communication
            config: Configuration dictionary
        """
        self.name = name
        self.bus = bus
        self.config = config

        self._thread: Optional[Thread] = None
        self._running = Event()
        self._rate_hz: Optional[float] = None

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize actor resources.

        Called once before the main loop starts.
        Subscribe to topics, initialize hardware, etc.
        """
        pass

    @abstractmethod
    def loop(self) -> None:
        """
        Main processing loop iteration.

        Called repeatedly at the configured rate (or as fast as possible).
        Read from subscribed queues, process, publish results.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Cleanup actor resources.

        Called once after the main loop stops.
        Unsubscribe, close connections, etc.
        """
        pass

    def set_rate(self, hz: float) -> None:
        """Set the loop rate in Hz."""
        self._rate_hz = hz

    def start(self) -> None:
        """Start the actor in a new thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._running.set()
        self._thread = Thread(
            target=self._run,
            name=self.name,
            daemon=True
        )
        self._thread.start()
        print(f"[{self.name}] Started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the actor and wait for thread to finish."""
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        print(f"[{self.name}] Stopped")

    def is_running(self) -> bool:
        """Check if actor is running."""
        return self._running.is_set()

    def _run(self) -> None:
        """Internal thread entry point."""
        try:
            self.setup()
        except Exception as e:
            print(f"[{self.name}] Setup error: {e}")
            self._running.clear()
            return

        period = 1.0 / self._rate_hz if self._rate_hz else None

        try:
            while self._running.is_set():
                t_start = time.time()

                try:
                    self.loop()
                except Exception as e:
                    print(f"[{self.name}] Loop error: {e}")

                # Rate limiting
                if period is not None:
                    elapsed = time.time() - t_start
                    if elapsed < period:
                        time.sleep(period - elapsed)
        finally:
            try:
                self.teardown()
            except Exception as e:
                print(f"[{self.name}] Teardown error: {e}")


class TimedActor(Actor):
    """Actor that runs at a fixed rate with timing diagnostics."""

    def __init__(self, name: str, bus: MessageBus, config: dict, rate_hz: float):
        super().__init__(name, bus, config)
        self.set_rate(rate_hz)
        self._loop_count = 0
        self._total_time = 0.0

    @property
    def loop_count(self) -> int:
        return self._loop_count

    @property
    def avg_loop_time_ms(self) -> float:
        ## its the average time taken per loop excluding sleep time
        if self._loop_count == 0:
            return 0.0
        return (self._total_time / self._loop_count) * 1000

    def _run(self) -> None:
        """Internal thread entry point with timing."""
        try:
            self.setup()
        except Exception as e:
            print(f"[{self.name}] Setup error: {e}")
            self._running.clear()
            return

        period = 1.0 / self._rate_hz if self._rate_hz else None

        try:
            while self._running.is_set():
                t_start = time.time()

                try:
                    self.loop()
                    self._loop_count += 1
                except Exception as e:
                    print(f"[{self.name}] Loop error: {e}")

                elapsed = time.time() - t_start
                self._total_time += elapsed

                # Rate limiting
                if period is not None and elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            try:
                self.teardown()
            except Exception as e:
                print(f"[{self.name}] Teardown error: {e}")
