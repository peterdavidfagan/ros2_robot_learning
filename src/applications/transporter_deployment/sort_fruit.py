"""A basic debug/poc of identifying fruits with dino and sorting them."""
import yaml
import numpy as np
import rclpy
from robot_workspaces.franka_table import FrankaTable 
from scipy.spatial.transform import Rotation as R

def sort_apples(config):
    """
    Demonstrate apple sorting with basic object detection.
    """
    env = FrankaTable(config)
    env.reset()
    domain_model = env.props_info

    for key, prop_info in domain_model.items():
        bbox = prop_info['bbox']
        pixel_coords = [(bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2]
        pick_pose = env.pixel_2_world(pixel_coords)
        quat = R.from_euler('xyz', [0, 180, 180], degrees=True).as_quat()
        pick_pose = np.concatenate([pick_pose, quat])
        
        pick_action = {
            'pose': pick_pose,
            'pixel_coords': pixel_coords,
            'gripper_rot': 0.0,
            }

        def sample_pose_in_rect(location, size):
            x, y, z = location
            length, width, height = size
            
            x_pos = np.random.uniform(x, x + length)
            y_pos = np.random.uniform(y, y + width)
            z_pos = np.random.uniform(z, z + height)

            return [x_pos, y_pos, z_pos]

        # TODO: detect target location instead of getting from config or use transporter to infer from demos
        if '_'.join(prop_info['prop_name'].split('_')[:-1]) == 'a_red_apple':
            place_pose = sample_pose_in_rect(
                **env.config['workspace']['target_locations']['a_red_apple']
                )
            quat = R.from_euler('xyz', [0, 180, 180], degrees=True).as_quat()
            place_pose = np.concatenate([place_pose, quat])  

            place_action = {
            'pose': place_pose, # randomly sample place target
            'pixel_coords': pixel_coords,
            'gripper_rot': 0.0,
            }
        
        elif '_'.join(prop_info['prop_name'].split('_')[:-1]) == 'a_green_apple':
            place_pose = sample_pose_in_rect(
                **env.config['workspace']['target_locations']['a_green_apple']
            )
            quat = R.from_euler('xyz', [0, 180, 180], degrees=True).as_quat()
            place_pose = np.concatenate([place_pose, quat])             
            place_action = {
            'pose': place_pose, # randomly sample place target
            'pixel_coords': pixel_coords,
            'gripper_rot': 0.0,
            }

        else:
            raise NotImplementedError

        env.step(pick_action)
        env.step(place_action)
        env.reset()

    env.close()

def main(args=None):
    with open('./config/franka_table.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    rclpy.init(args=args)
    while True: # we always want to be sorting apples :)
        sort_apples(config)
    
    rclpy.shutdown()

if __name__=="__main__":
    main()
