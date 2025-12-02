from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # ---- launch arguments ----
    launch_rviz   = LaunchConfiguration("launch_rviz")
    run_id        = LaunchConfiguration("run_id")
    camera_name   = LaunchConfiguration("camera_name")

    # hemisphere configuration (CLI configurable)
    hemi_radius        = LaunchConfiguration("hemi_radius")
    max_poses          = LaunchConfiguration("max_poses")
    hemi_num_latitudes = LaunchConfiguration("hemi_num_latitudes")
    hemi_points_per_lat = LaunchConfiguration("hemi_points_per_lat")

    declared_arguments = [
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz with MoveIt configuration",
        ),
        DeclareLaunchArgument(
            "run_id",
            default_value="run_001",
            description="ID of this capture run (e.g. run_001, run_002)",
        ),
        DeclareLaunchArgument(
            "camera_name",
            default_value="realsense",
            description="Logical camera name (e.g. realsense, zed_left, zed_right)",
        ),
        DeclareLaunchArgument(
            "hemi_radius",
            default_value="0.33",
            description="Hemisphere radius (meters)",
        ),
        DeclareLaunchArgument(
            "max_poses",
            default_value="100",
            description="Maximum number of poses to visit",
        ),
        DeclareLaunchArgument(
            "hemi_num_latitudes",
            default_value="12",
            description="Number of latitude bands for hemisphere sampling",
        ),
        DeclareLaunchArgument(
            "hemi_points_per_lat",
            default_value="20",
            description="Number of points per latitude band",
        ),
    ]

    # ---- MoveIt config ----
    moveit_config = (
        MoveItConfigsBuilder("ur10e_scene", package_name="my_ur10e_moveit_config")
        .to_moveit_configs()
    )

    # move_group node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # hemisphere + realsense capture node
    hemi_rs_node = Node(
        package="hemi_motion_rs",
        executable="hemi_motion_rs_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                # image topic coming from realsense2_camera
                "image_topic": "/camera/camera/color/image_raw",

                # structured saving layout
                "base_data_dir": "/root/ur_ws_sim/data",
                "run_id": run_id,
                "camera_name": camera_name,

                # hemisphere config from CLI
                "hemi_radius": hemi_radius,
                "max_poses": max_poses,
                "hemi_num_latitudes": hemi_num_latitudes,
                "hemi_points_per_lat": hemi_points_per_lat,
            },
        ],
    )

    # Realsense node
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        output="screen",
        parameters=[
            {
                "enable_color": True,
                "enable_depth": True,
                "rgb_camera.profile": "1280x720x30",
            }
        ],
    )

    # RViz (optional)
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
            realsense_node,
            move_group_node,
            hemi_rs_node,
            rviz_node,
        ]
    )

