from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # -------- Launch arguments --------
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")

    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10e",
            description="Type of UR robot (ur3e, ur5e, ur10e, etc.)",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="0.0.0.0",
            description="IP of the robot (0.0.0.0 when using fake hardware)",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description="Use fake hardware (simulation via ros2_control)",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Whether to launch RViz from the UR bringup",
        ),
    ]

    # Our custom scene xacro is in urdf/ur10e_scene.xacro in the package
    # ur_control.launch.py will prepend "urdf/" automatically
    description_file = "ur10e_scene.xacro"

    # Include standard UR bringup (driver + controllers)
    ur_bringup_share = FindPackageShare("ur_bringup")
    ur_control_launch = PathJoinSubstitution(
        [ur_bringup_share, "launch", "ur_control.launch.py"]
    )

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_control_launch),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": launch_rviz,
            # tell UR bringup to use our scene package + xacro
            "description_package": "my_ur10e_scene_description",
            "description_file": description_file,
        }.items(),
    )

    return LaunchDescription(declared_arguments + [ur_control])

