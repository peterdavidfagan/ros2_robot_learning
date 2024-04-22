#!/usr/bin/env python3
"""Demonstrating policy deployment for a langauge-conditioned policy."""

import rclpy
from rclpy.action import ActionClient
import numpy as np
from cv_bridge import CvBridge

from moveit.policies import Policy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import ServoCommandType
from ros2_dspy_msgs.action import QA
from std_srvs.srv import SetBool
from ament_index_python.packages import get_package_share_directory

import onnx
import onnxruntime as ort


from panda_policy_deployment_demos.panda_policy_deployment_demos_parameters import policy as params

class LanguageConditionedPolicy(Policy):
    """Dummy policy for testing."""

    def __init__(self):
        super().__init__(params)
        self._logger.info("Language-Conditioned policy initialized")
        
        # use CvBridge to convert sensor_msgs/Image to numpy array
        self.cv_bridge = CvBridge()
        
        # keep track of current language command
        self.current_command = None

        # start onnx inference session
        #model_path = get_package_share_directory("panda_policy_deployment_demos") + "/models/your_model.onnx"
        #self.cnn = ort.InferenceSession(model_path)

        # initialize servo service client
        self.servo_command_client = self.create_client(ServoCommandType, "/servo_node/switch_command_type")
        while not self.servo_command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.servo_pause_client = self.create_client(SetBool, "/servo_node/pause_servo")
        while not self.servo_pause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        # initialize llm action client
        self.llm_client = ActionClient(self, QA, "/ollama_predict")

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
        self.current_command = result

    def set_servo_command_type(self, command_type):
        """Sets the servo command type."""
        request = ServoCommandType.Request()
        request.command_type = command_type
        self.future = self.servo_command_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result():
            self.get_logger().info("Servo command type set to: %s" % command_type)
        else:
            self.get_logger().error("Failed to set servo command type: %s" % command_type)

    def forward(self, sensor_msg):
        """Forward pass of the policy."""
        if (self._is_active) and (self.current_command is not None):
            # convert image data to jax array
            cv_img = self.cv_bridge.imgmsg_to_cv2(sensor_msg, desired_encoding="rgb8")
            
            # combine image and langauge command
            #input_data = np.concatenate((cv_img, self.current_command), axis=0)

            # perform inference with onnx model
            #prediction = self.cnn.run(None, {"input": input_data})

            # For this demo, we will just publish random target poses
            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = "panda_link0"
            target.pose.position.x = np.random.uniform(0.3, 0.7)
            target.pose.position.y = np.random.uniform(-0.2, 0.2)
            target.pose.position.z = np.random.uniform(0.25, 0.55)
            target.pose.orientation.w = 0.0
            target.pose.orientation.x = 0.924
            target.pose.orientation.y = -0.382
            target.pose.orientation.z = 0.0
            self.command_pub.publish(target)
        else:
            self._logger.debug("Policy is not active")


def main():
    rclpy.init()
    logger = rclpy.logging.get_logger("policy_deployment")
    
    policy = LanguageConditionedPolicy()
    policy.active = True
    policy.set_servo_command_type(ServoCommandType.Request.POSE) 
    
    # try a basic language command
    policy.reason("How should I grasp a cup?")

    rclpy.spin(policy)
    policy.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
