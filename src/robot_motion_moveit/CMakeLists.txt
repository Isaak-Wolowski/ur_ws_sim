cmake_minimum_required(VERSION 3.8)
project(robot_motion_moveit)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(hemi_motion_moveit src/hemi_motion_moveit.cpp)

ament_target_dependencies(hemi_motion_moveit
  rclcpp
  geometry_msgs
  visualization_msgs
  moveit_ros_planning_interface
)

install(TARGETS hemi_motion_moveit
  DESTINATION lib/${PROJECT_NAME})

ament_package()
