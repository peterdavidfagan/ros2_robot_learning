import os
import sys
import argparse
import yaml
import math
import numpy as np
from ament_index_python.packages import get_package_share_directory

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata
import scipy.spatial.transform as st
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from cv_bridge import CvBridge
from moveit2_policy_msgs.action import Transporter
from sensor_msgs.msg import Image, CameraInfo
import message_filters

import cv2

from moveit2_data_collector.robot_workspaces.franka_table import FrankaTable
import envlogger
from envlogger.backends import tfds_backend_writer
import tensorflow as tf
import tensorflow_datasets as tfds


class TransporterActionClient(Node):

    def __init__(self):
        super().__init__('transporter_action_client')

        deployment_param_path = os.path.join(get_package_share_directory("panda_policy_deployment_demos"), "config", "transporter_deployment.yaml")
        with open(deployment_param_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        self.cv_bridge = CvBridge()
        self.action_client = ActionClient(self, Transporter, 'transporter')

        self.rgb=None
        self.depth=None
        self.goal_success=False

    def send_goal(self):
        # compose goal
        goal = Transporter.Goal()

        # request goal from action server
        self.action_client.wait_for_server()
        self.goal_future = self.action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        self.goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.success))
        self.goal_success = True

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.status))


def main(args=None):
    rclpy.init(args=args)
    action_client = TransporterActionClient()
    while True:
        action_client.send_goal()
        rclpy.spin_once(action_client)

if __name__ == '__main__':
    main()
    