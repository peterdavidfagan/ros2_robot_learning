import os
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

import launch_ros
from launch import LaunchDescription
from launch.actions import (
        OpaqueFunction, 
        IncludeLaunchDescription, 
        DeclareLaunchArgument, 
        RegisterEventHandler, 
        TimerAction, 
        ExecuteProcess)
from launch_ros.actions import Node, SetParameter
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import (
        PythonLaunchDescriptionSource, 
        load_python_launch_file_as_module
        )
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
  
    use_fake_hardware = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Use fake hardware",
    )

    # set up controller manager for robotiq gripper under namespace robotiq
    robotiq_xacro = os.path.join(
            get_package_share_directory("robotiq_description"),
            "urdf",
            "robotiq_2f_85_gripper.urdf.xacro",
            )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            robotiq_xacro,
            " ",
            "use_fake_hardware:=",
            LaunchConfiguration('use_fake_hardware'),
        ]
    )

    robotiq_description_param = {
        "robot_description": launch_ros.parameter_descriptions.ParameterValue(
            robot_description_content, value_type=str
        )
    }

    robotiq_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robotiq_description_param,
            os.path.join(get_package_share_directory("franka_robotiq_moveit_config"), "config", "robotiq_controllers.yaml",)
            ],
        output="both",
        namespace="robotiq",
        condition=UnlessCondition(LaunchConfiguration('use_fake_hardware'))
    )

    robotiq_control_node_mj = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robotiq_description_param,
            os.path.join(get_package_share_directory("franka_robotiq_moveit_config"), "config", "robotiq_controllers_mujoco.yaml",)
            ],
        output="both",
        namespace="robotiq",
        condition=IfCondition(LaunchConfiguration('use_fake_hardware'))
    )
    
    load_robotiq_controllers = []                                                                                                                                                                                                                                                                                                                                                                                                                                       
    robotiq_controllers = ['robotiq_gripper_controller','robotiq_state_broadcaster', 'robotiq_activation_controller']                                                                                                                                                                                                                                                                                                                                                   
    for controller in robotiq_controllers:                                                                                                                                                                                                                                                                                                                                                                                                                              
        if controller == 'robotiq_activation_controller':                                                                                                                                                                                                                                                                                                                                                                                                               
            load_robotiq_controllers += [                                                                                                                                                                                                                                                                                                                                                                                                                               
                ExecuteProcess(                                                                                                                                                                                                                                                                                                                                                                                                                                         
                    cmd=["ros2 run controller_manager spawner {} -c /robotiq/controller_manager".format(controller)],                                                                                                                                                                                                                                                                                                                                                   
                    shell=True,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    output="screen",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                    condition=UnlessCondition(LaunchConfiguration('use_fake_hardware'))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            ]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

            load_robotiq_controllers += [
                ExecuteProcess(
                    cmd=["ros2 run controller_manager spawner {} -c /robotiq/controller_manager".format(controller)],
                    shell=True,
                    output="screen",
                )
            ]

    return LaunchDescription(
        [
            use_fake_hardware,
            robotiq_control_node,
        ] + load_robotiq_controllers
        )
