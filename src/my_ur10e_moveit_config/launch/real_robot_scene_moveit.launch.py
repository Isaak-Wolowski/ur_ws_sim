from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # Whether to launch RViz from this file
    launch_rviz = LaunchConfiguration("launch_rviz")

    declared_arguments = [
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz with MoveIt configuration",
        ),
    ]

    # This uses the SAME MoveIt config you used in simulation.
    # It already knows about 'ur10e_scene', SRDF, kinematics, etc.
    moveit_config = (
        MoveItConfigsBuilder("ur10e_scene", package_name="my_ur10e_moveit_config")
        .to_moveit_configs()
    )

    # ---- move_group (planning + execution) ----
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # ---- hemisphere motion node ----
    hemi_node = Node(
        package="robot_motion_moveit",        # your package
        executable="hemi_motion_moveit",      # your node
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # ---- RViz with MoveIt plugin ----
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("my_ur10e_moveit_config"), "config", "moveit.rviz"]
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict()],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription(
        declared_arguments + [move_group_node, hemi_node, rviz_node]
    )

