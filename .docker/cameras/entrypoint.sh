#!/bin/bash

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --
source "$ROS_UNDERLAY/setup.bash" --

# rebuild zed wrapper as we may have updated configs
colcon build --packages-select zed_wrapper
source "$ROS_UNDERLAY/setup.bash" --


# excute docker command
exec "$@"
