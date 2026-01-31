#!/usr/bin/env python3
"""Test Kinova Gen3 robot connection."""

import sys
import argparse
import numpy as np

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2, Base_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

TCP_PORT = 10000
UDP_PORT = 10001


def test_connection(ip: str, username: str, password: str) -> bool:
    """
    Test robot connection and print status.

    Returns True if connection successful.
    """
    tcp_transport = None
    udp_transport = None
    tcp_session = None
    udp_session = None

    print(f"\n{'='*50}")
    print(f"Testing connection to Kinova Gen3")
    print(f"{'='*50}")
    print(f"IP: {ip}")
    print(f"TCP Port: {TCP_PORT}")
    print(f"UDP Port: {UDP_PORT}")
    print(f"{'='*50}\n")

    try:
        # TCP connection
        print("[1/4] Connecting via TCP...", end=" ", flush=True)
        tcp_transport = TCPTransport()
        tcp_router = RouterClient(tcp_transport, RouterClient.basicErrorCallback)
        tcp_transport.connect(ip, TCP_PORT)
        print("OK")

        # TCP session
        print("[2/4] Creating TCP session...", end=" ", flush=True)
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = username
        session_info.password = password
        session_info.session_inactivity_timeout = 10000
        session_info.connection_inactivity_timeout = 2000

        tcp_session = SessionManager(tcp_router)
        tcp_session.CreateSession(session_info)
        print("OK")

        # UDP connection
        print("[3/4] Connecting via UDP...", end=" ", flush=True)
        udp_transport = UDPTransport()
        udp_router = RouterClient(udp_transport, RouterClient.basicErrorCallback)
        udp_transport.connect(ip, UDP_PORT)

        udp_session = SessionManager(udp_router)
        udp_session.CreateSession(session_info)
        print("OK")

        # Read robot info
        print("[4/4] Reading robot state...", end=" ", flush=True)
        base = BaseClient(tcp_router)
        base_cyclic = BaseCyclicClient(udp_router)

        # Get arm state
        arm_state = base.GetArmState()
        feedback = base_cyclic.RefreshFeedback()
        print("OK")

        # Print robot info
        print(f"\n{'='*50}")
        print("CONNECTION SUCCESSFUL")
        print(f"{'='*50}")

        # Arm state
        state_names = {
            0: "UNSPECIFIED",
            1: "BASE_INITIALIZATION",
            2: "IDLE",
            3: "ARM_INITIALIZATION",
            4: "ARM_IN_FAULT",
            5: "ARM_MAINTENANCE",
            6: "ARM_SERVOING_LOW_LEVEL",
            7: "ARM_SERVOING_READY",
            8: "ARM_SERVOING_PLAYING_SEQUENCE",
            9: "ARM_SERVOING_MANUALLY_CONTROLLED",
        }
        state_name = state_names.get(arm_state.active_state, f"UNKNOWN ({arm_state.active_state})")
        print(f"\nArm State: {state_name}")

        # Joint positions
        positions = np.array([feedback.actuators[i].position for i in range(7)])
        print(f"\nJoint Positions (deg):")
        for i, pos in enumerate(positions):
            print(f"  Joint {i+1}: {pos:8.2f}")

        # Gripper
        try:
            gripper_req = Base_pb2.GripperRequest()
            gripper_req.mode = Base_pb2.GRIPPER_POSITION
            gripper = base.GetMeasuredGripperMovement(gripper_req)
            if gripper.finger:
                print(f"\nGripper Position: {gripper.finger[0].value:.2f} (0=open, 1=closed)")
        except Exception:
            print("\nGripper: Unable to read")

        print(f"\n{'='*50}\n")
        return True

    except Exception as e:
        print("FAILED")
        print(f"\n{'='*50}")
        print("CONNECTION FAILED")
        print(f"{'='*50}")
        print(f"\nError: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check robot is powered on")
        print(f"  2. Verify network connection (ping {ip})")
        print(f"  3. Check IP address is correct")
        print(f"  4. Verify credentials (default: admin/admin)")
        print(f"{'='*50}\n")
        return False

    finally:
        # Cleanup
        if udp_session:
            try:
                udp_session.CloseSession()
            except Exception:
                pass
        if tcp_session:
            try:
                tcp_session.CloseSession()
            except Exception:
                pass
        if udp_transport:
            try:
                udp_transport.disconnect()
            except Exception:
                pass
        if tcp_transport:
            try:
                tcp_transport.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Test Kinova Gen3 robot connection")
    parser.add_argument("--ip", default="192.168.1.10", help="Robot IP address")
    parser.add_argument("--username", default="admin", help="Username")
    parser.add_argument("--password", default="admin", help="Password")
    args = parser.parse_args()

    success = test_connection(args.ip, args.username, args.password)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
