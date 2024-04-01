#!/bin/bash

# install libfranka
sudo apt update && apt upgrade -y
sudo apt remove "*libfranka*" -y
sudo apt install -y  build-essential cmake git libpoco-dev libeigen3-dev
cd libfranka 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
cpack -G DEB
sudo dpkg -i libfranka*.deb

# source iron installation and install middleware
source /opt/ros/iron/setup.bash
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp -y

# install moveit dependencies
cd ./src/motion_planning
for repo in moveit2/moveit2.repos $(f="moveit2/moveit2_$ROS_DISTRO.repos"; test -r $f && echo $f); do vcs import < "$repo"; done

# install rosdep dependencies
cd ..
rosdep install --from-paths . --ignore-src --rosdistro iron -r -y
cd ..

# build the workspace
colcon build --packages-ignore libfranka libfranka-common
source ./install/setup.bash
