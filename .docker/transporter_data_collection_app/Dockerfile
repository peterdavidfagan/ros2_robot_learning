ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base
MAINTAINER Peter David Fagan "peterdavidfagan@gmail.com"

# set default DDS to Cyclone
ENV ROS_DISTRO=humble
ENV RMW_IMPLEMENTATION rmw_cyclonedds_cpp

# install system dependencies
RUN apt-get update && \
    apt-get install -y neovim libxcb-cursor0 libgmp-dev python3-pip && \
    pip install pyqt6 dm_env envlogger[tfds] scipy opencv-python cvbridge3 && \
    rm -rf /var/lib/apt/lists/*

# set up ROS workspace
ENV ROS_UNDERLAY /root/data_collection_ws/install
WORKDIR $ROS_UNDERLAY/..
COPY src/data_collection ./src/data_collection
COPY src/motion_planning ./src/motion_planning
COPY src/applications ./src/applications
COPY src/control/franka_ros2/franka_description ./src/control/franka_ros2/franka_description
COPY src/control/ros2_robotiq_gripper/robotiq_description ./src/control/ros2_robotiq_gripper/robotiq_description

RUN cd ./src/motion_planning && \
	for repo in moveit2/moveit2.repos $(f="moveit2/moveit2_rolling.repos"; test -r $f && echo $f); do vcs import < "$repo"; done

# set up workspace           
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    apt-get update && \
    apt-get install -y ros-${ROS_DISTRO}-rmw-cyclonedds-cpp && \
    rosdep update && \
    DEBIAN_FRONTEND=noninteractive \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro ${ROS_DISTRO} && \
    colcon build && \
    rm -rf /var/lib/apt/lists/*

# create entrypoint
COPY .docker/ros_entrypoint.sh /sbin/ros_entrypoint.sh
RUN sudo chmod 755 /sbin/ros_entrypoint.sh
ENTRYPOINT ["/sbin/ros_entrypoint.sh"]
