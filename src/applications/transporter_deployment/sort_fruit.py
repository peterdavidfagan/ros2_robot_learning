"""Application for sorting fruit objects."""
from robot_workspaces.franka_table import FrankaTable 


def sort_apples(config):
    """
    Demonstrate apple sorting with basic object detection.
    """
    env = FrankaTable(config)
    env.reset()
    domain_model = env.props_info()
    
    for prop in domain_model:
        pixel_coords = [prop.bbox[2]//2, prop.bbox[3]//2]
        pick_pose = env.pixel_to_world()
        
        pick_action = {
            'pose' pick_pose,
            'pixel_coords': pixel_coords,
            'gripper_rot': 0.0,
        }

        if prop.name == 'a red apple':
            # TODO: check if prop is within target
            place_action = {
            'pose' place_pose,
            'pixel_coords': pixel_coords,
            'gripper_rot': 0.0,
            }
        
        elif prop.name == 'a green apple':
            # TODO: check if prop is within target
            place_action = {
            'pose' place_pose,
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
    rclpy.init(args=args)
    while True: # we always want to be sorting apples :)
        sort_apples(args)
    rclpy.shutdown()

if __name__=="__main__":
    main()
    