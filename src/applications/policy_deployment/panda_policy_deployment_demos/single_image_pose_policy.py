#!/usr/bin/env python3
"""Demonstrating policy deployment for a policy that accepts a single image as input."""

import rclpy
import numpy as np
from cv_bridge import CvBridge

from moveit.policies import Policy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import ServoCommandType
from std_srvs.srv import SetBool
from ament_index_python.packages import get_package_share_directory

import onnx
import onnxruntime as ort

from panda_policy_deployment_demos.panda_policy_deployment_demos_parameters import policy as params

class SingleImagePolicy(Policy):
    """Dummy policy for testing."""

    def __init__(self):
        super().__init__(params)
        self._logger.info("Dummy policy initialized")
        
        # use CvBridge to convert sensor_msgs/Image to numpy array
        self.cv_bridge = CvBridge()
        
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
        if self._is_active:
            # convert image data to jax array
            cv_img = self.cv_bridge.imgmsg_to_cv2(sensor_msg, desired_encoding="rgb8")
            # perform inference with onnx model
            #prediction = self.cnn.run(None, {"input": cv_img})

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
    
    policy = SingleImagePolicy()
    policy.active = True
    policy.set_servo_command_type(ServoCommandType.Request.POSE) 
    
    rclpy.spin(policy)
    policy.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
