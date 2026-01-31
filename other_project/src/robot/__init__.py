"""Robot-specific code."""

# Lazy imports to avoid loading heavy dependencies (pinocchio, kortex_api)
# when only hardware interfaces are needed

def __getattr__(name):
    if name == "RobotModel":
        from .model import RobotModel
        return RobotModel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["RobotModel"]
