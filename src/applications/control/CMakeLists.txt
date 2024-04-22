cmake_minimum_required(VERSION 3.22)
project(panda_control_demos)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  src/pose_tracking_example.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY notebooks DESTINATION share/${PROJECT_NAME})

ament_package()
