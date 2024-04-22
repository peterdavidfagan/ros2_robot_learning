#!/usr/bin/env python3
"""
A ROS 2 calibration node.

Implementation is heavily inspired by the great work of Alexander Khazatsky.
Source: https://github.com/AlexanderKhazatsky/R2D2/blob/main/r2d2/calibration/calibration_utils.py

Helpful resources:
    - https://forum.opencv.org/t/eye-to-hand-calibration/5690
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from launch_param_builder import ParameterBuilder

import cv2
from cv2 import aruco
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import SetBool

from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory

class CameraCalibration(Node):
    """A ROS 2 node to calibrate camera"""

    def __init__(self):
        """Initialize"""
        super().__init__("camera_calibration")
        self.logger = self.get_logger()

        # ensure parallel execution of camera callbacks
        # we want to get images while also executing the calibration service
        self.camera_callback_group = ReentrantCallbackGroup()

        # set QOS profile for camera image callback
        self.camera_qos_profile = QoSProfile(
                depth=1,
                history=QoSHistoryPolicy(rclpy.qos.HistoryPolicy.KEEP_LAST),
                reliability=QoSReliabilityPolicy(rclpy.qos.ReliabilityPolicy.RELIABLE),
            )

        # read in calibration config
        self.calib_config = ParameterBuilder("panda_camera_calibration_demos").yaml("config/camera_calibration.yaml").to_dict()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
        self.charuco_board = cv2.aruco.CharucoBoard_create(
            self.calib_config["charuco"]["squares_x"],
            self.calib_config["charuco"]["squares_y"],
            self.calib_config["charuco"]["square_length"],
            self.calib_config["charuco"]["marker_length"],
            self.aruco_dict
        )
        self.detector_params = cv2.aruco.DetectorParameters_create()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH
        self.cv_bridge = CvBridge()

        # subscribe to camera info
        self.create_subscription(
            CameraInfo,
            self.calib_config["camera_info_topic"],
            self._camera_info_callback,
            10,
            callback_group=self.camera_callback_group
        )

        # subscribe to camera image
        self.create_subscription(
            Image,
            self.calib_config["camera_topic"],
            self._image_callback,
            self.camera_qos_profile,
            callback_group=self.camera_callback_group
        )
    
        # create a service client for running eye to hand calibration
        self.calib_client = self.create_service(
            SetBool,
            self.calib_config["eye_to_hand_calibration_service"],
            self.run_eye_2_hand_calibration,
        )

        # create a service client for running eye in hand calibration
        self.calib_client = self.create_service(
            SetBool,
            self.calib_config["eye_in_hand_calibration_service"],
            self.run_eye_in_hand_calibration,
        )

        # track latest values of camera info and image 
        self._latest_image = None
        self._camera_info = None

        # instantiate moveit python interface
        
        # TODO: move to node params
        robot_ip = "192.168.106.99"
        use_gripper = "true" 
        use_fake_hardware = "true" 

        moveit_config = (
            MoveItConfigsBuilder(robot_name="panda", package_name="franka_robotiq_moveit_config")
            .robot_description(file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/robot.urdf.xacro",
                mappings={
                    "robot_ip": robot_ip,
                    "robotiq_gripper": use_gripper,
                    "use_fake_hardware": use_fake_hardware,
                    })
            .robot_description_semantic("config/panda.srdf.xacro", 
                mappings={
                    "robotiq_gripper": use_gripper,
                    })
            .trajectory_execution("config/moveit_controllers.yaml")
            .moveit_cpp(
                file_path=get_package_share_directory("panda_motion_planning_demos")
                + "/config/moveit_cpp.yaml"
            )
            .to_moveit_configs()
            ).to_dict()

        self.panda = MoveItPy(config_dict=moveit_config)
        self.panda_arm = self.panda.get_planning_component("panda_arm")

    def _image_callback(self, msg):
        """Callback function for image topic"""
        self._latest_image = msg

    def _camera_info_callback(self, msg):
        """Callback function for camera info topic"""
        self._camera_info = msg
    
    def gripper2base(self):
        """Get the transform from the gripper coordinate frame to the base coordinate frame"""
        self.panda_arm.set_start_state_to_current_state()
        robot_state = self.panda_arm.get_start_state()
        return robot_state.get_frame_transform("panda_link8") # TODO: move to config
    
    def gripper_pose(self):
        """Get the pose of the gripper"""
        self.panda_arm.set_start_state_to_current_state()
        robot_state = self.panda_arm.get_start_state()
        
        pose = robot_state.get_pose("panda_link8") # TODO: move to config"

        pose_pos = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ])

        pos_euler = Rotation.from_quat([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]).as_euler("xyz", degrees=True)

        return np.concatenate([pose_pos, pos_euler])

    def move_to_random_pose(self, calibration_workspace):
        """Move the robot to a random pose within the workspace"""
        # set plan start state to current state
        self.panda_arm.set_start_state_to_current_state()
        
        # random pose within workspace
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "panda_link0" # TODO: move to config
        pose_goal.pose.position.x = np.random.uniform(
                calibration_workspace["x_min"], 
                calibration_workspace["x_max"]
                )
        pose_goal.pose.position.y = np.random.uniform(
                calibration_workspace["y_min"],
                calibration_workspace["y_max"]
                )
        pose_goal.pose.position.z = np.random.uniform(
                calibration_workspace["z_min"],
                calibration_workspace["z_max"]
                )
        
        # Create a rotation object from Euler angles specifying axes of rotation
        rot_x = np.random.uniform(
                calibration_workspace["rot_x_min"],
                calibration_workspace["rot_x_max"]
                )
        rot_y = np.random.uniform(
                calibration_workspace["rot_y_min"],
                calibration_workspace["rot_y_max"]
                )
        rot_z = np.random.uniform(
                calibration_workspace["rot_z_min"],
                calibration_workspace["rot_z_max"]
                )
        rot = Rotation.from_euler("XYZ", [rot_x, rot_y, rot_z], degrees=True)
        rot_quat = rot.as_quat()
        rot_eul = rot.as_euler("XYZ", degrees=True)
        
        pose_goal.pose.orientation.x = rot_quat[0]
        pose_goal.pose.orientation.y = rot_quat[1]
        pose_goal.pose.orientation.z = rot_quat[2]
        pose_goal.pose.orientation.w = rot_quat[3]
       
        # get current pose
        self.panda_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="panda_link8")
        
        # plan and execute
        plan_result = self.panda_arm.plan()
        if plan_result:
            robot_trajectory = plan_result.trajectory
            self.panda.execute(robot_trajectory, controllers=[])
            return True
        else:
            return False
    
    def visualize_image(self, image):
        """Visualize image"""
        self.logger.info("Visualizing image")
        cv2.imshow("Image", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


    def detect_charuco_board(self, image):
        """
        Detect charuco board in image

        Adapted from: https://github.com/AlexanderKhazatsky/R2D2/blob/1aa471ae35cd9b11e20cc004c15ad4c74e92605d/r2d2/calibration/calibration_utils.py#L122
        """
        # detect aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            image, 
            self.aruco_dict, 
            parameters=self.detector_params,
        )
        
        # find undetected markers
        corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
            image,
            self.charuco_board,
            corners,
            ids,
            rejectedImgPoints,
            parameters=self.detector_params,
            cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
            distCoeffs=np.array(self._camera_info.d),
            )

        # if no markers found, return
        if ids is None:
            return None, None

        # detect charuco board
        num_corners_found, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, 
            ids, 
            image, 
            self.charuco_board, 
            cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
            distCoeffs=np.array(self._camera_info.d),
        )

        # if no charuco board found, return
        if num_corners_found < 5:
            return None, None

        # draw detected charuco board
        image = aruco.drawDetectedCornersCharuco(
            image, charuco_corners,
        )

        # visualize image
        #self.visualize_image(image)

        return image, charuco_corners, charuco_ids, image.shape[:2]
    
    def calc_target_to_camera(self, readings):
        """
        Calculate target to camera transform

        Adapted from: https://github.com/AlexanderKhazatsky/R2D2/blob/1aa471ae35cd9b11e20cc004c15ad4c74e92605d/r2d2/calibration/calibration_utils.py#L164
        """
        init_corners_all = []  # Corners discovered in all images processed
        init_ids_all = []  # Aruco ids corresponding to corners discovered
        fixed_image_size = readings[0][3]

        # Proccess Readings #
        init_successes = []
        for i in range(len(readings)):
            corners, charuco_corners, charuco_ids, img_size = readings[i]
            assert img_size == fixed_image_size
            init_corners_all.append(charuco_corners)
            init_ids_all.append(charuco_ids)
            init_successes.append(i)

        # First Pass: Find Outliers #
        threshold = 10
        if len(init_successes) < threshold:
            return None

        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors = (
            aruco.calibrateCameraCharucoExtended(
                charucoCorners=init_corners_all,
                charucoIds=init_ids_all,
                board=self.charuco_board,
                imageSize=fixed_image_size,
                flags=self.calib_flags,
                cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
                distCoeffs=np.array(self._camera_info.d),
            )
        )

        # Remove Outliers #
        final_corners_all = [
                init_corners_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= 3.0 # TODO: read from params
        ]
        final_ids_all = [
            init_ids_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= 3.0
        ]
        final_successes = [
            init_successes[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= 3.0
        ]
        if len(final_successes) < threshold:
            return None

        # Second Pass: Calculate Finalized Extrinsics #
        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=final_corners_all,
            charucoIds=final_ids_all,
            board=self.charuco_board,
            imageSize=fixed_image_size,
            flags=self.calib_flags,
            cameraMatrix=np.array(self._camera_info.k).reshape(3,3),
            distCoeffs=np.array(self._camera_info.d),
        )
        
        self.logger.info("Calibration error: {}".format(calibration_error))
        self.logger.info("Camera matrix: {}".format(cameraMatrix))
        self.logger.info("Distortion coefficients: {}".format(distCoeffs))
        self.logger.info("Rotation vectors: {}".format(rvecs))
        self.logger.info("Translation vectors: {}".format(tvecs))

        # Return Transformation #
        if calibration_error > 3.0:
            return None

        rmats = [Rotation.from_rotvec(rvec.flatten()).as_matrix() for rvec in rvecs]
        tvecs = [tvec.flatten() for tvec in tvecs]

        return rmats, tvecs, final_successes

    def run_eye_2_hand_calibration(self, request, response):
        """Calibrate third person camera to robot base"""
        # check if we have both camera info and image
        if self._latest_image is None or self._camera_info is None:
            self.logger.warn("No image or camera info received yet")
            return

        # run calibration
        self.logger.info("Running calibration")
        
        # collect samples
        workspace_config = self.calib_config["eye_to_hand_calibration"]["workspace"]

        images = []
        gripper2base_vals = []
        gripper_poses = []
        for i in range(self.calib_config["eye_to_hand_calibration"]["num_samples"]):
            self.logger.info("Collecting sample {}".format(i))
            if self.move_to_random_pose(workspace_config):
                time.sleep(0.5) # required to ensure latest image is captured
                
                # capture image
                img = self.cv_bridge.imgmsg_to_cv2(self._latest_image, "mono8")
                images.append(img)
                #self.visualize_image(img)

                # capture gripper pose
                self.logger.info("Gripper pose: {}".format(self.gripper_pose()))
                gripper_poses.append(self.gripper_pose())

                # capture base to ee transform
                gripper2base = self.gripper2base()
                gripper2base_vals.append(gripper2base)
                
                # sleep
                time.sleep(self.calib_config["eye_to_hand_calibration"]["sample_delay"])
        
        # process captured images
        readings = []
        for image in images:
            #self.visualize_image(image)
            readings.append(self.detect_charuco_board(image))
        
        # calculate target to camera transform
        R_target2cam, t_target2cam, successes = self.calc_target_to_camera(readings)
        
        # filter gripper2base by successes
        gripper2base_vals = [gripper2base_vals[i] for i in successes]
        R_base2gripper = [t[:3,:3].T for t in gripper2base_vals]
        t_base2gripper = [-R @ t[:3,3] for R, t in zip(R_base2gripper, gripper2base_vals)]

        # run calibration for cam2base
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_base2gripper,
            t_gripper2base=t_base2gripper,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )
        
        # log results
        self.logger.info("Calibration results:")
        self.logger.info("Rotation matrix: {}".format(rmat))
        self.logger.info("Position: {}".format(pos))

        # return success response
        response.success = True
        response.message = "Calibration successful"
        return response

    def run_eye_in_hand_calibration(self, request, response):
        """Calibrate hand-mounted camera to robot gripper"""
        # check if we have both camera info and image
        if self._latest_image is None or self._camera_info is None:
            self.logger.warn("No image or camera info received yet")
            return

        # run calibration
        self.logger.info("Running calibration")
        
        # collect samples
        workspace_config = self.calib_config["eye_in_hand_calibration"]["workspace"]
        images = []
        gripper2base_vals = []
        gripper_poses = []
        for i in range(self.calib_config["eye_in_hand_calibration"]["num_samples"]):
            self.logger.info("Collecting sample {}".format(i))
            if self.move_to_random_pose(workspace_config):
                time.sleep(0.5) # required to ensure latest image is captured
                
                # capture image
                img = self.cv_bridge.imgmsg_to_cv2(self._latest_image, "mono8")
                images.append(img)
                #self.visualize_image(img)

                # capture gripper pose
                self.logger.info("Gripper pose: {}".format(self.gripper_pose()))
                gripper_poses.append(self.gripper_pose())

                # capture base to ee transform
                gripper2base = self.gripper2base()
                gripper2base_vals.append(gripper2base)
                
                # sleep
                time.sleep(self.calib_config["eye_in_hand_calibration"]["sample_delay"])
        
        # process captured images
        readings = []
        for image in images:
            #self.visualize_image(image)
            readings.append(self.detect_charuco_board(image))
        
        # calculate target to camera transform
        R_target2cam, t_target2cam, successes = self.calc_target_to_camera(readings)
        
        # filter gripper2base by successes
        gripper2base_vals = [gripper2base_vals[i] for i in successes]
        R_gripper2base = [t[:3,:3] for t in gripper2base_vals]
        t_gripper2base = [t[:3,3] for t in gripper2base_vals]

        # run calibration for cam2base
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )
        
        # log results
        self.logger.info("Calibration results:")
        self.logger.info("Rotation matrix: {}".format(rmat))
        self.logger.info("Position: {}".format(pos))

        # return success response
        response.success = True
        response.message = "Calibration successful"
        return response


if __name__=="__main__":
    rclpy.init()
    node = CameraCalibration()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()
