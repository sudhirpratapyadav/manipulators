"""
Kinova Gen3 hardware interface.

Handles TCP and UDP communication with the robot.
This is a thin wrapper - no control logic, just I/O.
"""

import time
import numpy as np
from typing import Optional, Tuple

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2, Base_pb2, BaseCyclic_pb2, ActuatorConfig_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient

from .hardware_base import HardwareInterface, RobotFeedback


class KinovaHardware(HardwareInterface):
    """
    Low-level hardware interface for Kinova Gen3.

    Responsibilities:
    - TCP connection for high-level commands
    - UDP connection for real-time feedback and torque commands
    - No control logic - just read/write operations
    """

    TCP_PORT = 10000
    UDP_PORT = 10001

    # Arm states we accept as "ready"
    READY_STATES = [Base_pb2.ARMSTATE_SERVOING_READY, 9]

    def __init__(self, ip: str, username: str, password: str):
        self.ip = ip
        self.username = username
        self.password = password

        # Connection state
        self._tcp_transport: Optional[TCPTransport] = None
        self._udp_transport: Optional[UDPTransport] = None
        self._tcp_router: Optional[RouterClient] = None
        self._udp_router: Optional[RouterClient] = None
        self._tcp_session: Optional[SessionManager] = None
        self._udp_session: Optional[SessionManager] = None

        # Kortex clients
        self.base: Optional[BaseClient] = None
        self.base_cyclic: Optional[BaseCyclicClient] = None
        self.actuator_config: Optional[ActuatorConfigClient] = None

        # UDP command structure (reused to avoid allocations)
        self._base_command: Optional[BaseCyclic_pb2.Command] = None
        self._send_option: Optional[RouterClientSendOptions] = None

        # Mode tracking
        self._in_torque_mode = False

    def connect(self) -> bool:
        """
        Establish TCP and UDP connections.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # TCP connection
            self._tcp_transport = TCPTransport()
            self._tcp_router = RouterClient(self._tcp_transport, RouterClient.basicErrorCallback)
            self._tcp_transport.connect(self.ip, self.TCP_PORT)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.username
            session_info.password = self.password
            session_info.session_inactivity_timeout = 10000
            session_info.connection_inactivity_timeout = 2000

            self._tcp_session = SessionManager(self._tcp_router)
            self._tcp_session.CreateSession(session_info)

            self.base = BaseClient(self._tcp_router)
            self.actuator_config = ActuatorConfigClient(self._tcp_router)

            # UDP connection
            self._udp_transport = UDPTransport()
            self._udp_router = RouterClient(self._udp_transport, RouterClient.basicErrorCallback)
            self._udp_transport.connect(self.ip, self.UDP_PORT)

            self._udp_session = SessionManager(self._udp_router)
            self._udp_session.CreateSession(session_info)

            self.base_cyclic = BaseCyclicClient(self._udp_router)

            # Initialize command structure
            self._init_command_structure()

            return True

        except Exception as e:
            print(f"[KinovaHardware] Connection error: {e}")
            self.disconnect()
            return False

    def disconnect(self) -> None:
        """Close all connections."""
        # Switch to position mode first
        if self._in_torque_mode:
            try:
                self.set_torque_mode(False)
            except Exception:
                pass

        # Close sessions
        router_options = RouterClientSendOptions()
        router_options.timeout_ms = 1000

        if self._udp_session:
            try:
                self._udp_session.CloseSession(router_options)
            except Exception:
                pass

        if self._tcp_session:
            try:
                self._tcp_session.CloseSession(router_options)
            except Exception:
                pass

        # Disconnect transports
        if self._udp_transport:
            try:
                self._udp_transport.disconnect()
            except Exception:
                pass

        if self._tcp_transport:
            try:
                self._tcp_transport.disconnect()
            except Exception:
                pass

        # Clear references
        self._tcp_transport = None
        self._udp_transport = None
        self._tcp_router = None
        self._udp_router = None
        self._tcp_session = None
        self._udp_session = None
        self.base = None
        self.base_cyclic = None
        self.actuator_config = None
        self._base_command = None
        self._in_torque_mode = False

    def _init_command_structure(self) -> None:
        """Initialize reusable command structure."""
        self._base_command = BaseCyclic_pb2.Command()
        for i in range(7):
            self._base_command.actuators.add()
            self._base_command.actuators[i].flags = 1
            self._base_command.actuators[i].command_id = 0
        self._base_command.frame_id = 0

        self._send_option = RouterClientSendOptions()
        self._send_option.andForget = False
        self._send_option.delay_ms = 0
        self._send_option.timeout_ms = 10

    def clear_faults(self) -> None:
        """Clear any robot faults."""
        if self.base:
            self.base.ClearFaults()

    def stop(self) -> None:
        """Stop all robot motion."""
        if self.base:
            try:
                self.base.Stop()
            except Exception:
                pass

    def set_servoing_mode(self, low_level: bool) -> None:
        """
        Set robot servoing mode.

        Args:
            low_level: True for LOW_LEVEL_SERVOING, False for SINGLE_LEVEL_SERVOING
        """
        if not self.base:
            return

        servo_mode = Base_pb2.ServoingModeInformation()
        if low_level:
            servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        else:
            servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(servo_mode)

    def set_torque_mode(self, enabled: bool) -> None:
        """
        Enable or disable torque control mode on all actuators.

        Args:
            enabled: True for TORQUE mode, False for POSITION mode
        """
        if not self.actuator_config:
            return

        control_mode_info = ActuatorConfig_pb2.ControlModeInformation()
        if enabled:
            control_mode_info.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
        else:
            control_mode_info.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')

        for i in range(7):
            self.actuator_config.SetControlMode(control_mode_info, i + 1)

        self._in_torque_mode = enabled

    def is_arm_ready(self) -> bool:
        """Check if arm is in ready state."""
        if not self.base:
            return False

        try:
            arm_state = self.base.GetArmState()
            return arm_state.active_state in self.READY_STATES
        except Exception:
            return False

    def wait_for_arm_ready(self, timeout: float = 10.0) -> bool:
        """Wait for arm to be in ready state."""
        start = time.time()
        while (time.time() - start) < timeout:
            if self.is_arm_ready():
                return True

            # Try clearing faults if in fault state
            try:
                arm_state = self.base.GetArmState()
                if arm_state.active_state == Base_pb2.ARMSTATE_IN_FAULT:
                    self.clear_faults()
            except Exception:
                pass

            time.sleep(0.1)
        return False

    def read_feedback(self) -> Optional[RobotFeedback]:
        """
        Read current robot state (non-blocking).

        Returns:
            RobotFeedback or None if error
        """
        if not self.base_cyclic:
            return None

        try:
            feedback = self.base_cyclic.RefreshFeedback()
            return RobotFeedback(
                positions_deg=np.array([feedback.actuators[i].position for i in range(7)]),
                velocities_deg=np.array([feedback.actuators[i].velocity for i in range(7)]),
                torques_measured=np.array([feedback.actuators[i].torque for i in range(7)]),
                timestamp=time.time(),
            )
        except Exception as e:
            print(f"[KinovaHardware] Read error: {e}")
            return None

    def send_torques(self, torques: np.ndarray, positions_deg: np.ndarray) -> Optional[RobotFeedback]:
        """
        Send torque command and read feedback.

        Args:
            torques: Joint torques (7,) in Nm
            positions_deg: Current positions for command echo (7,) in degrees

        Returns:
            RobotFeedback or None if error
        """
        if not self.base_cyclic or self._base_command is None:
            return None

        try:
            # Update command
            self._base_command.frame_id = (self._base_command.frame_id + 1) % 65536
            for i in range(7):
                self._base_command.actuators[i].position = positions_deg[i]
                self._base_command.actuators[i].torque_joint = torques[i]
                self._base_command.actuators[i].command_id = self._base_command.frame_id

            # Send and receive
            feedback = self.base_cyclic.Refresh(self._base_command, 0, self._send_option)

            return RobotFeedback(
                positions_deg=np.array([feedback.actuators[i].position for i in range(7)]),
                velocities_deg=np.array([feedback.actuators[i].velocity for i in range(7)]),
                torques_measured=np.array([feedback.actuators[i].torque for i in range(7)]),
                timestamp=time.time(),
            )
        except Exception as e:
            print(f"[KinovaHardware] Send error: {e}")
            return None

    def send_positions(self, positions_deg: np.ndarray) -> Optional[RobotFeedback]:
        """
        Send position command (for stabilization before torque mode).

        Args:
            positions_deg: Joint positions (7,) in degrees

        Returns:
            RobotFeedback or None if error
        """
        if not self.base_cyclic or self._base_command is None:
            return None

        try:
            self._base_command.frame_id = (self._base_command.frame_id + 1) % 65536
            for i in range(7):
                self._base_command.actuators[i].position = positions_deg[i]
                self._base_command.actuators[i].torque_joint = 0.0
                self._base_command.actuators[i].command_id = self._base_command.frame_id

            feedback = self.base_cyclic.Refresh(self._base_command, 0, self._send_option)

            return RobotFeedback(
                positions_deg=np.array([feedback.actuators[i].position for i in range(7)]),
                velocities_deg=np.array([feedback.actuators[i].velocity for i in range(7)]),
                torques_measured=np.array([feedback.actuators[i].torque for i in range(7)]),
                timestamp=time.time(),
            )
        except Exception:
            return None

    def execute_joint_action(
        self,
        target_positions_deg: np.ndarray,
        duration: float = 8.0,
        callback=None,
    ) -> bool:
        """
        Execute high-level joint position action.

        Args:
            target_positions_deg: Target joint angles (7,) in degrees
            duration: Motion duration in seconds
            callback: Optional callback for progress (not implemented)

        Returns:
            True if successful
        """
        if not self.base:
            return False

        try:
            action = Base_pb2.Action()
            action.name = "JointMove"
            action.reach_joint_angles.constraint.type = Base_pb2.JOINT_CONSTRAINT_DURATION
            action.reach_joint_angles.constraint.value = duration

            for i, angle in enumerate(target_positions_deg):
                joint = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint.joint_identifier = i
                joint.value = float(angle)

            # Setup notification
            action_done = [False]
            action_ok = [False]

            def on_action(notif):
                if notif.action_event == Base_pb2.ACTION_END:
                    action_done[0] = True
                    action_ok[0] = True
                elif notif.action_event == Base_pb2.ACTION_ABORT:
                    action_done[0] = True

            handle = self.base.OnNotificationActionTopic(on_action, Base_pb2.NotificationOptions())

            try:
                self.base.ExecuteAction(action)

                # Wait for completion
                timeout = duration + 10.0
                start = time.time()
                while not action_done[0] and (time.time() - start) < timeout:
                    time.sleep(0.1)
            finally:
                self.base.Unsubscribe(handle)

            return action_ok[0]

        except Exception as e:
            print(f"[KinovaHardware] Action error: {e}")
            return False

    @property
    def in_torque_mode(self) -> bool:
        return self._in_torque_mode
