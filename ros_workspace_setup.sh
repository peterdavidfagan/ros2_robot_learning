#!/bin/bash
set -e

# install libfranka
sudo apt update && apt upgrade -y || true
sudo apt remove "*libfranka*" -y || true
sudo apt install -y  build-essential cmake git libpoco-dev libeigen3-dev
cd libfranka 
if [ -d "./build" ]; then
	rm -rf build
fi
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
cpack -G DEB
sudo dpkg -i libfranka*.deb
cd ../..

# source iron installation and install middleware
source /opt/ros/humble/setup.bash
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp -y

# install moveit dependencies
if [ -d "./src/motion_planning" ]; then
	cd ./src/motion_planning
	for repo in moveit2/moveit2.repos $(f="moveit2/moveit2_rolling.repos"; test -r $f && echo $f); do vcs import < "$repo"; done
else
	echo "Ignoring motion planning dependencies"
fi

# install rosdep dependencies
cd ..
rosdep update
rosdep install --from-paths . --ignore-src --rosdistro $ROS_DISTRO -r -y
cd ..

# build the workspace
colcon build --packages-ignore libfranka libfranka-common
source ./install/setup.bash
