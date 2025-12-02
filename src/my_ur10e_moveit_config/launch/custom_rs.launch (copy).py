from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

from moveit_configs_utils import MoveItConfigsBuilder

import os


def generate_launch_description():
    # Whether to launch RViz from this file
    launch_rviz = LaunchConfiguration("launch_rviz")

    declared_arguments = [
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",  # RViz ON by default
            description="Launch RViz with MoveIt configuration",
        ),
    ]

    # Same MoveIt config as your normal hardware launch
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

    # ---- RealSense camera driver (rs_launch.py) ----
    realsense_launch_file = os.path.join(
        get_package_share_directory("realsense2_camera"),
        "launch",
        "rs_launch.py",
    )

    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_file),
        # you can add launch_arguments here if needed
    )

    # ---- hemisphere motion + RealSense capture node ----
    hemi_rs_node = Node(
        package="hemi_motion_rs",          # our new package
        executable="hemi_motion_rs_node",  # our node
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "image_topic": "/camera/camera/color/image_raw",
                "save_dir": "/root/ur_ws_sim/data/realsense_caps",
                # optional tunables:
                # "hemi_center_xyz": [0.0, 0.0, 0.30],
                # "hemi_radius": 0.33,
                # "max_poses": 80,
            },
        ],
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
        declared_arguments
        + [
            move_group_node,
            realsense_node,
            hemi_rs_node,
            rviz_node,
        ]
    )

