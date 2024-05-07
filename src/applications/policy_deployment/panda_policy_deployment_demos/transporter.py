#!/usr/bin/env python3
"""Demonstrating policy deployment for a policy that accepts a single image as input."""
import os
import argparse
import yaml

import numpy as np
import einops as e
import jax
import jax.numpy as jnp
import scipy.spatial.transform as st
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from matplotlib import cm

import message_filters
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import ServoCommandType
from moveit2_policy_msgs.action import Transporter
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from moveit2_data_collector.robot_workspaces.franka_table import FrankaTable
from panda_policy_deployment_demos.panda_policy_deployment_demos_parameters import policy as params


class TransporterActionServer(Node):
    """Policy deployment for transporter networks."""

    def __init__(self):
        super().__init__("transporter_action_client")

        # load deployment params
        deployment_param_path = os.path.join(get_package_share_directory("panda_policy_deployment_demos"), "config", "transporter_deployment.yaml")
        with open(deployment_param_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        # subscribers for synchronized image feed
        self.sensor_subs = []        
        self.sensor_subs.append(
            message_filters.Subscriber(
                self,
                Image,
                self.config['camera']['image_topic'],
            )
        )
        self.sensor_subs.append(
            message_filters.Subscriber(
                self,
                Image,
                self.config['camera']['depth_topic'],
            )
        )
        self.sensor_sync = message_filters.ApproximateTimeSynchronizer(
            self.sensor_subs,
            5,
            0.5,
        )
        self.sensor_sync.registerCallback(self.update_image_data)

        # publishers to monitor transporter predictions
        self.pick_prediction_publisher = self.create_publisher(Image, 'pick_qvals', 10)
        self.place_prediction_publisher = self.create_publisher(Image, 'place_qvals', 10)
        
        # use CvBridge to convert sensor_msgs/Image to numpy array
        self.cv_bridge = CvBridge()
        
        # download model files from hugging face
        hf_hub_download(
            repo_id="peterdavidfagan/transporter_networks",
            repo_type="model",
            filename="transporter_pick.onnx",
            local_dir="/tmp/models",
        )
        
        hf_hub_download(
            repo_id="peterdavidfagan/transporter_networks",
            repo_type="model",
            filename="transporter_place.onnx",
            local_dir="/tmp/models",
        )
        
        # start onnx inference session
        self.pick_model = ort.InferenceSession("/tmp/models/transporter_pick.onnx")
        self.place_model = ort.InferenceSession("/tmp/models/transporter_place.onnx")

        # initialize workspace environment
        self.env = FrankaTable()

        # instantiate action server
        self.action_server = ActionServer(
            self,
            Transporter,
            'transporter',
            self.forward,
            )
        
        # assign camera intrinsics
        fx = self.config["camera"]["intrinsics"]["fx"]
        fy = self.config["camera"]["intrinsics"]["fy"]
        cx = self.config["camera"]["intrinsics"]["cx"]
        cy = self.config["camera"]["intrinsics"]["cy"]
        self.camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # assign camera extrinsics
        translation = [
            self.config["camera"]["extrinsics"]["x"],
            self.config["camera"]["extrinsics"]["y"],
            self.config["camera"]["extrinsics"]["z"],
            ]
        quaternion = [
            self.config["camera"]["extrinsics"]["qx"],
            self.config["camera"]["extrinsics"]["qy"],
            self.config["camera"]["extrinsics"]["qz"],
            self.config["camera"]["extrinsics"]["qw"],
            ]
        rotation = st.Rotation.from_quat(quaternion).as_matrix()
        self.camera_extrinsics = np.eye(4)
        self.camera_extrinsics[:3, :3] = rotation
        self.camera_extrinsics[:3, 3] = translation

        # workspace image crop
        self.u_min = self.config["camera"]["crop"]["top_left_u"]
        self.u_max = self.config["camera"]["crop"]["bottom_right_u"]
        self.v_min = self.config["camera"]["crop"]["top_left_v"]
        self.v_max = self.config["camera"]["crop"]["bottom_right_v"]

    def update_image_data(self, rgb, depth):
        self.rgb = rgb
        self.depth = depth

    def forward(self, goal_handle):
        """Predict action with transporter network and execute with MoveIt."""
        
        # convert ros messages to cv2
        rgb = self.cv_bridge.imgmsg_to_cv2(self.rgb, "rgb8")
        depth = self.cv_bridge.imgmsg_to_cv2(self.depth, "32FC1")

        # crop images about workspace
        rgb_crop_raw = jax.lax.slice(rgb, (self.v_min, self.u_min, 0), (self.v_max, self.u_max, 3))
        depth_crop_raw = jax.lax.slice(depth, (self.v_min, self.u_min), (self.v_max, self.u_max))

        # display images for debug purposes
        # plt.imshow(rgb_crop_raw)
        # plt.show(block=True)

        # process depth
        nan_mask = jnp.isnan(depth_crop_raw)
        inf_mask = jnp.isinf(depth_crop_raw)
        mask = jnp.logical_or(nan_mask, inf_mask)
        max_val = jnp.max(depth_crop_raw, initial=0, where=~mask)
        depth_crop_filled = jnp.where(~mask, depth_crop_raw, max_val) # for now fill with max_val and hope the q-network learns to compensate

        # normalize and concatenate
        rgb_crop = jax.nn.standardize(rgb_crop_raw / 255.0)
        depth_crop = jax.nn.standardize(depth_crop_filled)
        depth_crop = e.rearrange(depth_crop, "h w -> h w 1")
        rgbd_crop, _ = e.pack([rgb_crop, depth_crop], 'h w *')

        # perform inference with transporter network
        rgbd_input = np.expand_dims(rgbd_crop.__array__().astype(np.float64), axis=0)
        pick_q_vals = self.pick_model.run(None, {"rgbd": rgbd_input})[0][0]
        pick_pixels = np.unravel_index(np.argmax(pick_q_vals.reshape(360, 360)), (360,360))
        #print(f"pick_pixels: {pick_pixels}")

        # crop image about pick
        u_min = jnp.max(jnp.asarray([0, pick_pixels[1]-50]))
        v_min = jnp.max(jnp.asarray([0, pick_pixels[0]-50]))
        rgbd_pick_crop = jax.lax.dynamic_slice(
            rgbd_crop, 
            (u_min, v_min, 0), 
            (100, 100, 4)
            )
        rgbd_crop_input = np.expand_dims(rgbd_pick_crop.__array__().astype(np.float64), axis=0)

        place_q_vals = self.place_model.run(None, {"rgbd": rgbd_input, "rgbd_crop": rgbd_crop_input})[0][0]
        place_pixels = np.unravel_index(np.argmax(place_q_vals.reshape(360, 360)), (360,360))
        #print(f"place pixels: {place_pixels}")

        # publish predictions
        self.pick_prediction_publisher.publish(self.q_vals_2_img(pick_q_vals))
        self.place_prediction_publisher.publish(self.q_vals_2_img(place_q_vals))

        # execute actions
        pick_action_dict = {
            "pose": self.pixel_2_world(pick_pixels),
            "pixel_coords": pick_pixels,
            "gripper_rot": 0, # defined wrt base frame, note z-axis of gripper frame points in direction of grasp
        }

        place_action_dict = {
            "pose": self.pixel_2_world(place_pixels), 
            "pixel_coords": place_pixels,
            "gripper_rot": 0, # defined wrt base frame, note z-axis of gripper frame points in direction of grasp
        }

        # execute action using MoveIt in robot workspace
        self.env.step(pick_action_dict)
        self.env.step(place_action_dict)

        # return result of action execution
        result = {'success': True}

        return result

    def pixel_2_world(self, coords):
        """
        Converts pixel coord in inference image to real-world coordinates.
        """
        
        depth_img = self.cv_bridge.imgmsg_to_cv2(self.depth, "32FC1")
        
        # start by inpainting depth values (sometimes sensor returns nan/inf)
        nan_mask = np.isnan(depth_img)
        inf_mask = np.isinf(depth_img)
        mask = np.logical_or(nan_mask, inf_mask)
        mask = cv2.UMat(mask.astype(np.uint8))
        scale = np.ma.masked_invalid(np.abs(depth_img)).max() # scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_img = depth_img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)

        # interpolate remaining nan values with nearest neighbor
        depth_img = np.array(depth_img.get())
        y, x = np.where(~np.isnan(depth_img))
        x_range, y_range = np.meshgrid(np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0]))
        depth_img = griddata((x, y), depth_img[y, x], (x_range, y_range), method='nearest')
        depth_img = depth_img * scale 

        # convert pixel coordinates to coordinates in raw image
        v = coords[0] + self.v_min
        u = coords[1] + self.u_min
        depth_val = depth_img[v, u]
        #print(f"v: {v}, u: {u}")

        # convert current pixels coordinates to camera frame coordinates
        pixel_coords = np.array([u, v])
        image_coords = np.concatenate([pixel_coords, np.ones(1)])
        camera_coords =  np.linalg.inv(self.camera_intrinsics) @ image_coords
        camera_coords *= depth_val # negate depth when using mujoco camera convention

        # convert camera coordinates to world coordinates
        camera_coords = np.concatenate([camera_coords, np.ones(1)])
        world_coords = self.camera_extrinsics @ camera_coords
        world_coords = world_coords[:3] / world_coords[3]
        quat = R.from_euler('xyz', [0, 180, 180], degrees=True).as_quat() # TODO: update when also predicting gripper rotation
        pose = np.concatenate([world_coords, quat])
        print(pose)

        return pose
       
    def q_vals_2_img(self, q_vals):
        #q_vals = (q_vals - q_vals.min()) / ((q_vals.max() - q_vals.min()))
        #pick_heatmap = q_vals.reshape((360, 360))
        #plt.imshow(np.asarray(cm.viridis(pick_heatmap)*255, dtype=np.uint8))
        #plt.show(block=True)
        
        normalized_heatmap = cv2.normalize(q_vals.reshape(360, 360), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

        return self.cv_bridge.cv2_to_imgmsg(colormap, encoding="bgr8")

def main():
    rclpy.init()
    transporter_action_server = TransporterActionServer()
    rclpy.spin(transporter_action_server)
    transporter_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
