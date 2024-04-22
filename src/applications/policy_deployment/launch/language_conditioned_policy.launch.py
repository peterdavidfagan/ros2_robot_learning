from launch import LaunchDescription
from launch_param_builder import ParameterBuilder
from launch_ros.actions import Node


def generate_launch_description():
    
    policy_params = {
            "policy_node": ParameterBuilder("panda_policy_deployment_demos")
            .yaml("config/policy.yaml")
            .to_dict()
            }

    policy_node = Node(
            package="panda_policy_deployment_demos",
            executable="language_conditioned_policy",
            parameters=[policy_params],
            output="both",
            )

    return LaunchDescription([policy_node])
