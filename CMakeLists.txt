cmake_minimum_required(VERSION 3.8)
project(cuda_demo)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86")

set(CUDA_SOURCES
  src/vector_add.cu
)

cuda_add_library(cuda_kernel STATIC ${CUDA_SOURCES})

add_executable(cuda_demo_node src/cuda_demo_node.cpp)

target_link_libraries(cuda_demo_node
  cuda_kernel
  rclcpp::rclcpp
)

ament_target_dependencies(cuda_demo_node
  rclcpp
)

install(TARGETS
  cuda_demo_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
