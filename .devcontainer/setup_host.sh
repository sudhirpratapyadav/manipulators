#!/bin/bash
# Run this script on the HOST machine (not inside the container)
# to set up RealSense camera udev rules.

set -e

echo "Installing RealSense udev rules..."
sudo curl -sSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules -o /etc/udev/rules.d/99-realsense-libusb.rules

echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "Done! Please unplug and replug the RealSense camera."
echo "Then rebuild the dev container: Ctrl+Shift+P -> 'Dev Containers: Rebuild Container'"
