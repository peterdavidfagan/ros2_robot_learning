#!/bin/bash

# setup python environment
source .venv/bin/activate

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --
source "$ROS_UNDERLAY/setup.bash" --

# excute docker command
exec "$@"
