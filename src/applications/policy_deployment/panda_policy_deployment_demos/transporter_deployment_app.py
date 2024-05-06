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

from robot_workspaces.franka_table import FrankaTable
import envlogger
from envlogger.backends import tfds_backend_writer
import tensorflow as tf
import tensorflow_datasets as tfds


class TransporterActionClient(Node):

    def __init__(self):
        super().__init__('transporter action client')

        deployment_param_path = os.path.join(get_package_share_directory("panda_policy_deployment_demos"), "config", "transporter_deployment.yaml")
        with open(deployment_param_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        self.cv_bridge = CvBridge()

        # create message filter for image topics
        self.sensor_subs = []        
        self.sensor_subs.append(
            message_filters.Subscriber(
                self,
                Image,
                self.config['camera']['image_topic'],
                10,
            )
        )
        self.sensor_subs.append(
            message_filters.Subscriber(
                self,
                Image,
                self.config['camera']['depth_topic'],
                10,
            )
        )
        self.sensor_sync = message_filters.ApproximateTimeSynchronizer(
            self.sensor_subs,
            5,
            0.5,
        )
        self.sensor_sync.registerCallback(self.update_image_data)

        # create action client
        self.action_client = ActionClient(self, Transporter, 'transporter')


    def update_image_data(self, rgb, depth):
        self.rgb = rgb
        self.depth = depth

    def send_goal(self):
        # compose goal
        goal = Transporter.Goal()
        goal.rgb = self.rgb
        goal.depth = self.depth 

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

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.status))


def main(args=None):
    rclpy.init(args=args)
    action_client = TransporterActionClient()
    while True:
        future = action_client.send_goal()
        rclpy.spin_until_future_complete(action_client, future)

if __name__ == '__main__':
    main()
    