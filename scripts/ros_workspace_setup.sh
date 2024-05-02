#!/bin/bash

# parse input args
# Initialize variables with default values
poetry_build=true

# Function to display usage information
usage() {
    echo "Usage: $0 [--poetry_build <true/false>]"
    echo "Options:"
    echo "  --poetry_build         Specify whether to build poetry env or not (optional, default: false)"
    exit 1
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --poetry_build)
        case $2 in
            true)
            poetry_build=true
            ;;
            false)
            poetry_build=false
            ;;
            *)
            echo "Invalid value for --poetry_build. Please use 'true' or 'false'."
            usage
            ;;
        esac
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown option: $key"
        usage
        ;;
    esac
done

set -e

# enter root directory
cd ..

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

# install and activate python environment
if $poetry_build; then
	[ -e poetry.lock ] && rm poetry.lock
	poetry install
	poetry shell
else
	source .venv/bin/activate
fi

# source humble installation and install middleware
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
