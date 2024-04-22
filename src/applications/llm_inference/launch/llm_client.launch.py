from launch import LaunchDescription
from launch_param_builder import ParameterBuilder
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart

def generate_launch_description():
    
    # launch ollama inference server
    start_ollama_server = ExecuteProcess(
        cmd=["ros2", "launch", "ros2_dspy", "ollama_inference.launch.py"],
        output="screen"
    )

    llm_client_node = Node(
            package="panda_llm_inference_demos",
            executable="llm_client.py",
            output="both",
            )

    return LaunchDescription([
            start_ollama_server,
            RegisterEventHandler(
                OnProcessStart(
                    target_action=start_ollama_server,
                    on_start=[
                        TimerAction(
                            period=20.0,
                            actions=[llm_client_node]
                            )]
                    )
                )
            ])
