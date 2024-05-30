"""
A basic dm environment for deploying the transporter network on the Franka robot.
"""
import time
from typing import Dict
from copy import deepcopy
from ament_index_python.packages import get_package_share_directory

import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as st
from scipy.interpolate import griddata
import dm_env
from cv_bridge import CvBridge
import cv2

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.logging import get_logger
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import message_filters

from moveit.planning import MoveItPy, PlanRequestParameters, MultiPipelinePlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder

from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import Image, CameraInfo
from ros2_object_detection_msgs.srv import DetectObjectGroundedDino


def plan_and_execute(
    robot,
    planning_component,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        raise RuntimeError("Failed to plan trajectory")

    time.sleep(sleep_time)


class GripperClient(Node):

    def __init__(self, gripper_controller):
        super().__init__("gripper_client")
        self.gripper_action_client = ActionClient(
            self,
            GripperCommand, 
            gripper_controller,
        )
    
    def close_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.8
        goal.command.max_effort = 5.0
        self.gripper_action_client.wait_for_server()
        return self.gripper_action_client.send_goal_async(goal)

    def open_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.0
        goal.command.max_effort = 5.0
        self.gripper_action_client.wait_for_server()
        return self.gripper_action_client.send_goal_async(goal)


class FrankaTable(Node, dm_env.Environment):
    """
    This dm_env for deploying transporter network applications.
    """

    def __init__(self, config):
        super().__init__("transporter_deployment")
        self.logger = self.get_logger()
        self.config = config 
        self.robotiq_tcp_z_offset = config['robot']['gripper_tcp_offset']
        self.dino_confidence_threshold = config['grounded_dino']['confidence_threshold']
        self.dino_prompts = config['grounded_dino']['prompts']
        
        # set up motion planning client
        moveit_config = (
            MoveItConfigsBuilder(robot_name="panda", package_name="franka_robotiq_moveit_config")
            .robot_description(file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/robot.urdf.xacro",
                mappings={
                    "robot_ip": self.config['robot']['ip'],
                    "robotiq_gripper": self.config['robot']['use_gripper'],
                    "use_fake_hardware": self.config['robot']['use_fake_hardware'],
                    })
            .robot_description_semantic("config/panda.srdf.xacro", 
                mappings={
                    "robotiq_gripper": self.config['robot']['use_gripper'],
                    })
            .trajectory_execution("config/moveit_controllers.yaml")
            .moveit_cpp(
                file_path=get_package_share_directory("panda_motion_planning_demos")
                + "/config/moveit_cpp_mujoco.yaml"
            )
            .to_moveit_configs()
            ).to_dict()
        
        # create clients for moveit motion planning
        self.panda = MoveItPy(config_dict=moveit_config)
        self.planning_scene_monitor = self.panda.get_planning_scene_monitor()
        self.panda_arm = self.panda.get_planning_component("panda_arm") 
        self.gripper_client = GripperClient(self.config['robot']['gripper_controller'])

        # add ground plane collision geometry to moveit planning scene
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "panda_link0"
            collision_object.id = "ground_plane"

            box_pose = Pose()
            box_pose.position.x = 0.0
            box_pose.position.y = 0.0
            box_pose.position.z = 0.0

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [2.0, 2.0, 0.001]

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD

            scene.apply_collision_object(collision_object)
       
            # finally handle the allowed collisions for the object
            scene.allowed_collision_matrix.set_entry("ground_plane", "panda_link0", True)
            scene.allowed_collision_matrix.set_entry("ground_plane", "panda_link1", True)
            scene.allowed_collision_matrix.set_entry("ground_plane", "robotiq_85_left_finger_tip_link", True)
            scene.allowed_collision_matrix.set_entry("ground_plane", "robotiq_85_right_finger_tip_link", True)

            scene.current_state.update()  # Important to ensure the scene is updated

        # instantiate grounded dino client
        self.grounded_dino_detection_client = self.create_client(DetectObjectGroundedDino, 'grounded_dino_detect_object')

        # set up camera 
        self.cv_bridge = CvBridge()
        self.camera_callback_group = ReentrantCallbackGroup()
        self.camera_qos_profile = QoSProfile(
                depth=1,
                history=QoSHistoryPolicy(rclpy.qos.HistoryPolicy.KEEP_LAST),
                reliability=QoSReliabilityPolicy(rclpy.qos.ReliabilityPolicy.RELIABLE),
            )

        self.rgb_image_sub = message_filters.Subscriber(
            self,
            Image,
            self.config['camera']['rgb_topic'],
            callback_group=self.camera_callback_group,
        )

        self.depth_image_sub = message_filters.Subscriber(
            self,
            Image,
            self.config['camera']['depth_topic'],
            callback_group=self.camera_callback_group,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_image_sub, self.depth_image_sub],
            10,
            0.1,
            )
        self.sync.registerCallback(self.image_callback)

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
        # self.camera_extrinsics[:3, :3] = rotation
        # self.camera_extrinsics[:3, 3] = translation
        self.camera_extrinsics[:3, :3] = rotation
        self.camera_extrinsics[:3, 3] = translation
        # TODO: load transporter models for pick/place inference

        self.mode="pick"
        self.current_observation = None

    def reset(self) -> dm_env.TimeStep:        
        # return to home state
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(configuration_name="ready")
        plan_and_execute(self.panda, self.panda_arm, sleep_time=1.0)
        
        # open gripper
        self.gripper_client.open_gripper()
        self.mode = "pick"

        return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=0.0,
                discount=0.0,
                observation=deepcopy(self.current_observation),
                )

    def step(self, action_dict) -> dm_env.TimeStep:
        if self.mode == "pick":
            self.pick(action_dict["pose"])
        else:
            self.place(action_dict["pose"])

        return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=0.0,
                discount=0.0,
                observation=deepcopy(self.current_observation),
                )

    def set_observation(self, rgb, depth):
        self.current_observation = {
            "overhead_camera/rgb": rgb,
            "overhead_camera/depth": depth,
        }

    def set_metadata(self, config):
        self.metadata = config

    def observation_spec(self) -> Dict[str, dm_env.specs.Array]:
        return {
                "overhead_camera/rgb": dm_env.specs.Array(shape=(621,1104, 3), dtype=np.float32),
                "overhead_camera/depth": dm_env.specs.Array(shape=(621, 1104), dtype=np.float32),
                }

    def action_spec(self) -> dm_env.specs.Array:
        return {
                "pose": dm_env.specs.Array(shape=(7,), dtype=np.float64), # [x, y, z, qx, qy, qz, qw]
                "pixel_coords": dm_env.specs.Array(shape=(2,), dtype=np.int64), # [u, v]
                "gripper_rot": dm_env.specs.Array(shape=(1,), dtype=np.float64),
                }
    
    def reward_spec(self) -> dm_env.specs.Array:
        return dm_env.specs.Array(
                shape=(),
                dtype=np.float64,
                )

    def discount_spec(self) -> dm_env.specs.Array:
        return dm_env.specs.Array(
                shape=(),
                dtype=np.float64,
                )

    def close(self):
        print("closing")

    def image_callback(self, rgb, depth):
        self.rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb, "rgb8")
        self.depth_img = self.cv_bridge.imgmsg_to_cv2(depth, "32FC1") # check encoding

    def pixel_2_world(self, coords):
        """
        Note: first coordinate must by u, second v.
        """
        depth_img = self.depth_img.copy()
        u = coords[0] 
        v = coords[1]
        depth_val = depth_img[v, u] # indexing change as v corresponds to rows, and u columns

        # convert current pixels coordinates to camera frame coordinates
        pixel_coords = np.array([u, v])
        image_coords = np.concatenate([pixel_coords, np.ones(1)])
        camera_coords =  np.linalg.inv(self.camera_intrinsics) @ image_coords
        camera_coords *= depth_val
        #camera_coords *= -depth_val # negate depth when using mujoco camera convention

        # convert camera coordinates to world coordinates
        camera_coords = np.concatenate([camera_coords, np.ones(1)])
        world_coords = self.camera_extrinsics @ camera_coords
        world_coords = world_coords[:3] / world_coords[3]

        return world_coords

    def request_grounded_dino_detections(self, confidence_threshold, prompt):
        self.logger.info("requesting dino detections")

        # formulate the request
        request = DetectObjectGroundedDino.Request()
        request.confidence_threshold = confidence_threshold
        request.prompt = prompt
        
        # send the request
        self.future = self.grounded_dino_detection_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        self.logger.info("finished dino detection")
        return self.future.result()

    # TODO: add camera intrinsics/extrinsics here

    @property
    def props_info(self) -> dict:
        """       
        Parses detected objects into a domain model, that is a dictionary of objects and their properties.
        """
        props_info = {}
        for prompt in self.dino_prompts:
            # get detected objects from grounded dino
            self.detections = self.request_grounded_dino_detections(self.dino_confidence_threshold, prompt).bounding_boxes

            # for each detection store bounding box and label
            for idx, bbox in enumerate(self.detections):
                info = {}
                info['prop_name'] = f"{bbox.object}_{idx}"
                info['bbox'] = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                info['symbols'] = [bbox.object]
                props_info[idx] = info

        return props_info

    def pick(self, pose):
        multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
            self.panda, ["pilz_lin", "pilz_ptp", "ompl_rrtc"]
        )

        pick_pose_msg = PoseStamped()
        pick_pose_msg.header.frame_id = "panda_link0"
        pick_pose_msg.pose.position.x = pose[0]
        pick_pose_msg.pose.position.y = pose[1]
        pick_pose_msg.pose.position.z = pose[2] + self.robotiq_tcp_z_offset
        pick_pose_msg.pose.orientation.x = pose[3]
        pick_pose_msg.pose.orientation.y = pose[4]
        pick_pose_msg.pose.orientation.z = pose[5]
        pick_pose_msg.pose.orientation.w = pose[6]
        
        self.pick_height=pose[2] # set this variable so it can be referenced in place motion

        # prepick pose
        self.panda_arm.set_start_state_to_current_state()
        pre_pick_pose_msg = deepcopy(pick_pose_msg)
        pre_pick_pose_msg.pose.position.z = 0.6
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_pick_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # pick pose
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(pose_stamped_msg=pick_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # close gripper
        self.gripper_client.close_gripper()
        time.sleep(3.0)
        
        # prepick arm
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_pick_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # go to ready position
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(configuration_name="ready")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # switch mode to place
        self.mode = "place"

    def place(self, pose):
        multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
            self.panda, ["pilz_lin", "pilz_ptp", "ompl_rrtc"]
        )

        place_pose_msg = PoseStamped()
        place_pose_msg.header.frame_id = "panda_link0"
        place_pose_msg.pose.position.x = pose[0]
        place_pose_msg.pose.position.y = pose[1]
        place_pose_msg.pose.position.z = self.pick_height + self.robotiq_tcp_z_offset
        place_pose_msg.pose.orientation.x = pose[3]
        place_pose_msg.pose.orientation.y = pose[4]
        place_pose_msg.pose.orientation.z = pose[5]
        place_pose_msg.pose.orientation.w = pose[6]
        
        # preplace pose
        self.panda_arm.set_start_state_to_current_state()
        pre_place_pose_msg = deepcopy(place_pose_msg)
        pre_place_pose_msg.pose.position.z = 0.6
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_place_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # place pose
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(pose_stamped_msg=place_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # open gripper
        self.gripper_client.open_gripper()
        time.sleep(3.0)
        
        # preplace arm
        self.panda_arm.set_start_state_to_current_state()
        pre_place_pose_msg = deepcopy(place_pose_msg)
        pre_place_pose_msg.pose.position.z = 0.6
        self.panda_arm.set_goal_state(pose_stamped_msg=pre_place_pose_msg, pose_link="panda_link8")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # go to ready position
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(configuration_name="ready")
        plan_and_execute(self.panda, self.panda_arm, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=0.5)

        # switch mode to pick
        self.mode = "pick"
