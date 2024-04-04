[![docs](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/actions/workflows/pages.yaml/badge.svg)](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/blob/franka_emika_panda/.github/workflows/pages.yaml)[![control](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/actions/workflows/control.yaml/badge.svg)](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/blob/franka_emika_panda/.github/workflows/control.yaml) [![motion_planning](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/actions/workflows/motion_planning.yaml/badge.svg)](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/blob/franka_emika_panda/.github/workflows/motion_planning.yaml) [![zed](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/actions/workflows/zed.yaml/badge.svg)](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/blob/ufactory_lite6/.github/workflows/zed.yaml)
[![foxglove_bridge](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/actions/workflows/foxglove_bridge.yaml/badge.svg)](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit/blob/ufactory_lite6/.github/workflows/foxglove_bridge.yaml)

# ROS 2 Robot Learning Workspace 
This toolkit provides a set of ROS packages with examples of using these packages in robot learning research. An official release of this workspace will be published for the Franka Robotics Panda robot in the coming weeks at which point all demos will be tested and verified for fixed versions of Franka Robotics firmware. 

<img src="./assets/robotics_toolkit.jpeg" height=300/>

[**[Documentation]**](https://peterdavidfagan.com/ros2_robotics_research_toolkit/) &ensp;


# Supported Franka Robotics Versions

| OS | ROS 2 Version | Franka Robot | Franka System Version | Libfranka Version |
| --- | --- | --- | --- | --- |
| Ubuntu 20.04 | Humble | Panda | 4.2.2 | custom backport of latest libfranka |


# Docker Support
In order to avoid users having to manage the installation and building of this ROS workspace I am releasing Docker containers which should function across various Ubuntu operating systems. Where the host system requires specific dependencies or settings (e.g. realtime kernel patch) I provide a setup shell script to configure the host system, further details to be added to the official documentation.
