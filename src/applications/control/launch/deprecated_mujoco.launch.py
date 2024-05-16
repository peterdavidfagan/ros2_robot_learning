"""
A launch file for running the motion planning python api tutorial
"""
import os
import pathlib
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder

FILE_PATH = pathlib.Path(__file__).parent.absolute()
# TODO: use get package share directory
NOTEBOOK_DIR = get_package_share_directory("panda_control_demos") + "/notebooks"

def generate_launch_description():
    
    # declare parameter for using robot ip
    robot_ip = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.106.99",
        description="Robot IP",
    )

    # declare parameter for using gripper
    use_gripper = DeclareLaunchArgument(
        "use_gripper",
        default_value="true",
        description="Use gripper",
    )
    
    # declare parameter for using fake controller
    use_fake_hardware = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Use fake hardware",
    )


    start_foxglove_bridge = ExecuteProcess(
            cmd=["ros2", "launch", "foxglove_bridge", "foxglove_bridge_launch.xml"],
            output="screen",
            )
    
    # for some reason with docker its quite slow to stream data
    #start_foxglove_studio = ExecuteProcess(
    #        cmd=["docker", "compose", "-f", DOCKER_COMPOSE_FILE_PATH, "up"],
    #        output="screen",
    #        )
    
    #open_foxglove_studio = ExecuteProcess(
    #        cmd=["xdg-open", "http://localhost:8080"],
    #        output="screen",
    #        )

    start_mujoco_sim = ExecuteProcess(
            cmd=["ros2", "launch", "mujoco_ros", "mjsim.launch.py"],
            output="screen",
            )

    start_ros_2_controllers = ExecuteProcess(
        cmd=["ros2", "launch", "panda_control_demos", "control_server.launch.py"],
        output="screen",
        )
    
    start_moveit_prerequisites = ExecuteProcess(
        cmd=["ros2", "launch", "panda_motion_planning_demos", "motion_planning_prerequisites.launch.py"],
        output="screen",
        )

    start_moveit_servo = ExecuteProcess(
        cmd=["ros2", "launch", "panda_control_demos", "servo_pose_jtc.launch.py"],
        output="screen",
        )

    start_control_demo = ExecuteProcess(
        cmd=["ros2", "run", "panda_control_demos", "pose_tracking_example.py"],
        output="screen",
        )

    return LaunchDescription(
        [
            # parameters
            robot_ip,
            use_gripper,
            use_fake_hardware,
            
            # launching processes
            start_mujoco_sim,
            RegisterEventHandler(
                OnProcessStart(
                    target_action=start_mujoco_sim,
                    on_start=[
                        TimerAction(
                            period=5.0,
                            actions=[
                                start_foxglove_bridge,
                                ]
                            )
                        ]
                    )
                ),

            RegisterEventHandler(
                OnProcessStart(
                    target_action=start_foxglove_bridge,
                    on_start=[
                        TimerAction(
                            period=10.0, 
                            actions=[
                                start_ros_2_controllers,
                                start_moveit_prerequisites,
                                start_moveit_servo,
                                ]
                            ),
                        ]
                )
            ),

            RegisterEventHandler(
                OnProcessStart(
                    target_action=start_moveit_servo,
                    on_start=[
                        TimerAction(
                            period=1.0,
                            actions=[
                                start_control_demo,
                                ]
                            )
                        ]
                    )
                )
        ]
        )
