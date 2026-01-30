

2. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv build-essential
   ```

6. **Build the ROS2 workspace:**
   ```bash
   cd ros2_ws
   colcon build
   source install/setup.bash
   cd ..
   ```

8. **Environment Setup:**
   For each new terminal session, you'll need to source the environment:
   ```bash
   source venv/bin/activate
   source /opt/ros/humble/setup.bash  # Adjust for your ROS2 distribution
   source ros2_ws/install/setup.bash
   ```


## Claude Code installation
```
sudo apt-get update
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code --prefix ~/.npm-global
echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc

curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc 
```