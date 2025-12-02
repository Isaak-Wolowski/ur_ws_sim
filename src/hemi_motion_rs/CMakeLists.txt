cmake_minimum_required(VERSION 3.8)
project(hemi_motion_rs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(hemi_motion_rs_node src/hemi_motion_rs_node.cpp)

ament_target_dependencies(hemi_motion_rs_node
  rclcpp
  geometry_msgs
  visualization_msgs
  sensor_msgs
  cv_bridge
  image_transport
  moveit_ros_planning_interface
)

target_link_libraries(hemi_motion_rs_node
  ${OpenCV_LIBRARIES}
)

install(TARGETS
  hemi_motion_rs_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

