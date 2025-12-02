from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Build MoveIt config from your config package
    moveit_config = (
        MoveItConfigsBuilder("ur10e_scene", package_name="my_ur10e_moveit_config")
        .to_moveit_configs()
    )

    # move_group node (planning + controllers)
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # hemisphere node: give it the SAME robot_description + SRDF + planning params
    hemi_node = Node(
        package="robot_motion_moveit",        # <- your package
        executable="hemi_motion_moveit",      # <- your node executable
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([move_group_node, hemi_node])

