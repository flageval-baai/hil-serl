#!/bin/bash

# Kill existing franka_server.py processes with the same arguments
echo "Killing existing franka_server.py processes..."
ps -ef | grep 'serl_robot_infra/robot_servers/franka_server.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Give it a moment to terminate
sleep 1

# Source the setup.bash file for the second ROS workspace
echo "Sourcing ROS workspace..."
source /home/baai/code/catkin_ws/devel/setup.bash

# Change the ROS master URI to a different port
export ROS_MASTER_URI=http://localhost:11311

# Run the new instance in the foreground
echo "Starting new franka_server.py in the foreground..."
python serl_robot_infra/robot_servers/franka_server.py \
    --robot_ip=172.16.0.3 \
    --gripper_ip=0 \
    --gripper_type=Robotiq \
    --reset_joint_target=0,0,0,-1.9,-0,2,0 \
    --flask_url=127.0.0.2 \
    --ros_port=11311
