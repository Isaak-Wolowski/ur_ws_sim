from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # -------- Launch arguments --------
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    launch_rviz = LaunchConfiguration("launch_rviz")

    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10e",
            description="Type of UR robot (ur3e, ur5e, ur10e, etc.)",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.2",  # your real robot IP
            description="IP of the real robot",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Whether to launch RViz from the UR driver",
        ),
    ]

    # Our custom scene xacro (robot + cart + checkerboard + camera)
    # NOTE: ur_control.launch.py will automatically prepend 'urdf/'.
    description_file = "ur10e_scene.xacro"

    # Use the NEW recommended package: ur_robot_driver
    ur_driver_share = FindPackageShare("ur_robot_driver")
    ur_control_launch = PathJoinSubstitution(
        [ur_driver_share, "launch", "ur_control.launch.py"]
    )

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_control_launch),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "use_fake_hardware": "false",              # real robot
            "launch_rviz": launch_rviz,
            # point the driver at your scene URDF
            "description_package": "my_ur10e_scene_description",
            "description_file": description_file,
        }.items(),
    )

    return LaunchDescription(declared_arguments + [ur_control])

