cmake_minimum_required(VERSION 3.8)
project(my_ur10e_scene_description)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Install urdf and (optionally) meshes
install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# If you later add meshes/, uncomment:
 install(
   DIRECTORY meshes
   DESTINATION share/${PROJECT_NAME}
 )

ament_package()
