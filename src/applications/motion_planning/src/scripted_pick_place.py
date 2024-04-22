#!/usr/bin/env python3
"""
A script to outline the fundamentals of the moveit_py motion planning API.
"""

import time
from copy import deepcopy

# generic ros libraries
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.logging import get_logger

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from geometry_msgs.msg import PoseStamped
from control_msgs.action import GripperCommand


# define gripper action client
class GripperClient(Node):

    def __init__(self):
        super().__init__("gripper_client")
        self.gripper_action_client = ActionClient(
            self,
            GripperCommand, 
            "/robotiq_position_controller/gripper_cmd"
        )
    
    def close_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.8
        goal.command.max_effort = 3.0
        self.gripper_action_client.wait_for_server()
        return self.gripper_action_client.send_goal_async(goal)

    def open_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.0
        goal.command.max_effort = 3.0
        self.gripper_action_client.wait_for_server()
        return self.gripper_action_client.send_goal_async(goal)


def plan_and_execute(
    robot,
    planning_component,
    logger,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    logger.info("Planning trajectory")
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
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        logger.info("Trajectory: {}".format(robot_trajectory))
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)

def pick_and_place_block(
        panda,
        panda_arm,
        logger,
        gripper_client,
        pick_pose,
        place_pose,
        ):
    """Helper function to pick and place a block."""
    # convert poses to PoseStamped messages
    pick_pose_msg = PoseStamped()
    pick_pose_msg.header.frame_id = "panda_link0"
    pick_pose_msg.pose.orientation.x = 0.9238795
    pick_pose_msg.pose.orientation.y = -0.3826834
    pick_pose_msg.pose.orientation.z = 0.0
    pick_pose_msg.pose.orientation.w = 0.0
    pick_pose_msg.pose.position.x = pick_pose[0]
    pick_pose_msg.pose.position.y = pick_pose[1]
    pick_pose_msg.pose.position.z = pick_pose[2]
    
    place_pose_msg = PoseStamped()
    place_pose_msg.header.frame_id = "panda_link0"
    place_pose_msg.pose.orientation.x = 0.9238795
    place_pose_msg.pose.orientation.y = -0.3826834
    place_pose_msg.pose.orientation.z = 0.0
    place_pose_msg.pose.orientation.w = 0.0
    place_pose_msg.pose.position.x = place_pose[0]
    place_pose_msg.pose.position.y = place_pose[1]
    place_pose_msg.pose.position.z = place_pose[2]

    
    # prepick pose
    logger.info("Going to prepick pose")
    panda_arm.set_start_state_to_current_state()
    pre_pick_pose_msg = deepcopy(pick_pose_msg)
    pre_pick_pose_msg.pose.position.z += 0.1
    panda_arm.set_goal_state(pose_stamped_msg=pre_pick_pose_msg, pose_link="panda_link8")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    # pick pose
    logger.info("Going to pick pose")
    panda_arm.set_start_state_to_current_state()
    panda_arm.set_goal_state(pose_stamped_msg=pick_pose_msg, pose_link="panda_link8")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    # close gripper
    logger.info("Closing gripper")
    gripper_client.close_gripper()
    time.sleep(2.0)
    
    # raise arm
    logger.info("Raising arm")
    panda_arm.set_start_state_to_current_state()
    pre_pick_pose_msg.pose.position.z += 0.2
    panda_arm.set_goal_state(pose_stamped_msg=pre_pick_pose_msg, pose_link="panda_link8")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    # preplace pose
    logger.info("Going to preplace pose")
    panda_arm.set_start_state_to_current_state()
    pre_place_pose_msg = deepcopy(place_pose_msg)
    pre_place_pose_msg.pose.position.z += 0.1
    panda_arm.set_goal_state(pose_stamped_msg=pre_place_pose_msg, pose_link="panda_link8")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    # place pose
    logger.info("Going to place pose")
    panda_arm.set_start_state_to_current_state()
    panda_arm.set_goal_state(pose_stamped_msg=place_pose_msg, pose_link="panda_link8")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    # open gripper
    logger.info("Opening gripper")
    gripper_client.open_gripper()
    time.sleep(2.0)

    # raise arm
    logger.info("Raising arm")
    panda_arm.set_start_state_to_current_state()
    pre_place_pose_msg.pose.position.z += 0.2
    panda_arm.set_goal_state(pose_stamped_msg=pre_place_pose_msg, pose_link="panda_link8")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

def main():
    
    # for this demo example we will hard code the poses
    height_adjustment = 0.175
    BLOCK1_POSE = [0.4251066, 0.0881298, height_adjustment]
    BLOCK2_POSE = [0.59047847, -0.07463033, height_adjustment]
    BLOCK3_POSE = [0.33900857, -0.19225322, height_adjustment]
    PLACE1_POSE = [0.5, 0.0, height_adjustment]
    PLACE2_POSE = PLACE1_POSE.copy()
    PLACE2_POSE[-1] += 0.08
    PLACE3_POSE = PLACE1_POSE.copy()
    PLACE3_POSE[-1] += 0.13

    rclpy.init()
    logger = get_logger("moveit_py.pose_goal")

    panda = MoveItPy(node_name="moveit_py")
    panda_arm = panda.get_planning_component("panda_arm")
    gripper_client = GripperClient()
    logger.info("MoveItPy instance created")

    pick_and_place_block(
        panda,
        panda_arm,
        logger,
        gripper_client,
        pick_pose=BLOCK1_POSE,
        place_pose=PLACE1_POSE,
    )

    pick_and_place_block(
        panda,
        panda_arm,
        logger,
        gripper_client,
        pick_pose=BLOCK2_POSE,
        place_pose=PLACE2_POSE,
    )

    pick_and_place_block(
        panda,
        panda_arm,
        logger,
        gripper_client,
        pick_pose=BLOCK3_POSE,
        place_pose=PLACE3_POSE,
    )


if __name__ == "__main__":
    main()
