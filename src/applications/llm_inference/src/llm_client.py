#!/usr/bin/env python3
"""Demonstrating policy deployment for a langauge-conditioned policy."""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import ServoCommandType
from ros2_dspy_msgs.action import QA
from std_srvs.srv import SetBool
from ament_index_python.packages import get_package_share_directory

import onnx
import onnxruntime as ort


from panda_policy_deployment_demos.panda_policy_deployment_demos_parameters import policy as params

class LLMClient(Node):
    """Dummy policy for testing."""

    def __init__(self):
        super().__init__("llm_client")
        self._logger.info("Language-Conditioned policy initialized")
        
        # initialize llm action client
        self.llm_client = ActionClient(self, QA, "/ollama_predict")
        
        # store current response
        self.current_response = None

    def reason(self, language_command):
        """Reason about the language command."""
        # construct question
        QA_goal = QA.Goal()
        QA_goal.question = language_command
        self._logger.info(f"Asking: {language_command}")

        # send question to llm and reset current command based on response
        self.llm_client.wait_for_server()
        self.goal_future = self.llm_client.send_goal_async(QA_goal)
        self.goal_future.add_done_callback(self.reason_response_callback)

    def reason_response_callback(self, future):
        """Callback for the reason response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._logger.error("Goal rejected :(")
            return

        self._logger.info("Goal accepted")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.reason_result_callback)

    def reason_result_callback(self, future):
        """Callback for the reason result."""
        result = future.result().result
        self.logger.info(f"Reasoned command: {result}")
        self.current_response = result


def main():
    rclpy.init()
    logger = rclpy.logging.get_logger("policy_deployment")
    
    # try a basic language command
    client = LLMClient()
    client.reason("How should I grasp a cup?")
    response = client.current_response

    client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
