#!/usr/bin/env python3
"""Demonstrating policy deployment for a policy that accepts a single image as input."""

import yaml
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge

from moveit2_data_collector.robot_workspaces.franka_table import FrankaTable

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import ServoCommandType
from moveit2_policy_msgs.action import Transporter
from std_srvs.srv import SetBool

import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from panda_policy_deployment_demos.panda_policy_deployment_demos_parameters import policy as params


class TransporterActionServer(Node):
    """Policy deployment for transporter networks."""

    def __init__(self):
        super().__init__(params)
        self._logger.info("Initializing transporter policy.")
        
        # use CvBridge to convert sensor_msgs/Image to numpy array
        self.cv_bridge = CvBridge()
        
        # download model file from hugging face
        hf_hub_download(
        repo_id="peterdavidfagan/transporter_networks",
        repo_type="model",
        filename="transporter.onnx",
        local_dir="/tmp/models/transporter.onnx",
        )

        # start onnx inference session
        self.model = ort.InferenceSession("/tmp/models/transporter.onnx")

        # initialize workspace environment
        self.env = FrankaTable() # TODO: pass proper args here

        # instantiate action server
        self.action_server = ActionServer(
            self,
            Transporter,
            'transporter',
            self.forward,
            )

        # TODO: store camera configuration params
        with open('placeholder.yml', 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
        
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


    def forward(self, goal_handle):
        """Predict action with transporter network and execute with MoveIt."""
        
        # get input data
        rgb = goal_handle.request.rgb
        depth = goal_handle.request.depth
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb, "rgb8")
        depth = self.cv_bridge.imgmsg_to_cv2(depth, "32FC1")

        # crop images about workspace
        # TODO: read crop indices from config
        rgb_crop_raw = jax.lax.slice(rgb, (crop_idx[0], crop_idx[1], 0), (crop_idx[2], crop_idx[3],3))
        depth_crop_raw = jax.lax.slice(depth, (crop_idx[0], crop_idx[1]), (crop_idx[2], crop_idx[3]))

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
        results = self.model.run(["pick_qvals", "place_qvals"], {"rgbd": rgbd_crop})

        pick_action_dict = {
            "pose": self.pixel_2_world(results['pick_qvals'], depth_data),
            "pixel_coords": np.array([u, v]),
            "gripper_rot": 180, # defined wrt base frame, note z-axis of gripper frame points in direction of grasp
        }

        place_action_dict = {
            "pose": self.pixel_2_world(results['place_qvals'], depth_data), 
            "pixel_coords": np.array([u, v]),
            "gripper_rot": 180, # defined wrt base frame, note z-axis of gripper frame points in direction of grasp
        }

        # execute action using MoveIt in robot workspace
        self.env.step(pick_action_dict)
        self.env.step(place_action_dict)

        # return result of action execution
        result = {'success': True}

        return result

    def pixel_2_world(self, coords, depth_data):
        """
        Converts pixel coord in inference image to real-world coordinates.
        """
        
        # convert pixel coordinates to coordinates in raw image


        # convert current pixels coordinates to camera frame coordinates
        pixel_coords = np.array([u, v])
        image_coords = np.concatenate([pixel_coords, np.ones(1)])
        camera_coords =  np.linalg.inv(self.camera_intrinsics) @ image_coords
        camera_coords *= depth_data[] # negate depth when using mujoco camera convention

        # convert camera coordinates to world coordinates
        camera_coords = np.concatenate([camera_coords, np.ones(1)])
        world_coords = self.camera_extrinsics @ camera_coords
        world_coords = world_coords[:3] / world_coords[3]
        quat = R.from_euler('xyz', [0, 180, 180], degrees=True).as_quat() # TODO: update when also predicting gripper rotation
        pose = np.concatenate([world_coords, quat])

        return pose
       

def main():
    rclpy.init()
    transporter_action_server = TransporterActionServer()
    rclpy.spin(transporter_action_server)
    transporter_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
