from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hemi_motion_rs',
            executable='hemi_motion_rs_node',
            name='hemi_motion_rs_node',
            output='screen',
            parameters=[{
                'move_group': 'manipulator',
                'ik_frame': 'tool0',
                'target_frame': 'checkerboard_link',
                'hemi_center_xyz': [0.0, 0.0, 0.30],
                'hemi_radius': 0.33,
                'hemi_num_latitudes': 12,
                'hemi_points_per_lat': 20,
                'axis_yaw_deg': 0.0,
                'max_poses': 100,
                'image_topic': '/camera/camera/color/image_raw',
                'save_dir': '/root/ur_ws_sim/data/realsense_caps',
            }]
        )
    ])

