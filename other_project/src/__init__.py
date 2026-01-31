"""
Kinova Gen3 Controller - Message-Passing Actor Architecture

This is a complete rewrite using a message-passing actor model for better:
- Testability (each actor can be tested independently)
- Extensibility (add new inputs/control modes via plugins)
- Maintainability (clear data flow, no shared mutable state)
- Performance (minimal overhead ~1-2μs per message)
"""

__version__ = "2.0.0"
