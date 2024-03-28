#!/bin/bash

# update submodules
git submodule sync
git submodule update --init --recursive

# install moveit dependencies
cd ./src/motion_planning
for repo in moveit2/moveit2.repos $(f="moveit2/moveit2_$ROS_DISTRO.repos"; test -r $f && echo $f); do vcs import < "$repo"; done

# install rosdep dependencies
cd ..
rosdep install --from-paths . --ignore-src --rosdistro rolling -r -y
cd ..

# build the workspace
source /opt/ros/rolling/setup.bash
colcon build --mixin release
source ./install/setup.bash
