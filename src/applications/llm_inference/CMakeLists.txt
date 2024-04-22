cmake_minimum_required(VERSION 3.22)
project(panda_llm_inference_demos)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  src/llm_client.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
