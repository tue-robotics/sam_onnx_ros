cmake_minimum_required(VERSION 3.5)

project(sam_onnx_ros)

add_compile_options(-Wall -Werror=all)
add_compile_options(-Wextra -Werror=extra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(console_bridge REQUIRED)
find_package(OpenCV REQUIRED)

option(${PROJECT_NAME}_ENABLE_TENSORRT "Build the optional SpeedSAM/TensorRT backend when TensorRT is available" ON)
if(DEFINED SAM_ONNX_ROS_ENABLE_TENSORRT)
  set(${PROJECT_NAME}_ENABLE_TENSORRT ${SAM_ONNX_ROS_ENABLE_TENSORRT})
endif()

set(SAM_ONNX_ROS_CATKIN_COMPONENTS
  onnxruntime_ros
)
set(SAM_ONNX_ROS_CATKIN_DEPENDS
  onnxruntime_ros
)
set(SAM_ONNX_ROS_PUBLIC_INCLUDE_DIRS
  include
)

if(${PROJECT_NAME}_ENABLE_TENSORRT)
  set(SAM_ONNX_ROS_TENSORRT_ENABLED TRUE)
  list(APPEND SAM_ONNX_ROS_CATKIN_COMPONENTS tensorrt_ros)
  list(APPEND SAM_ONNX_ROS_CATKIN_DEPENDS tensorrt_ros)
  list(APPEND SAM_ONNX_ROS_PUBLIC_INCLUDE_DIRS 3rdparty/sam_trt_lib/include)
else()
  set(SAM_ONNX_ROS_TENSORRT_ENABLED FALSE)
endif()

find_package(catkin REQUIRED
  COMPONENTS
  ${SAM_ONNX_ROS_CATKIN_COMPONENTS}
)

set(${PROJECT_NAME}_CUDA_ENABLED ${onnxruntime_ros_CUDA_ENABLED})
if(${PROJECT_NAME}_CUDA_ENABLED OR SAM_ONNX_ROS_TENSORRT_ENABLED)
  find_package(CUDAToolkit REQUIRED)
  set(${PROJECT_NAME}_CUDA_CATKIN_DEPENDS "CUDAToolkit")
  set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
  set(${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES "CUDA::cudart")
endif()

set(SAM_ONNX_ROS_CONFIG_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/config.hpp)
# @ONLY changes substitution behavior from: default: replace both ${VAR} and @VAR@ with @ONLY: replace only @VAR@
configure_file(include/${PROJECT_NAME}/config.hpp.in ${SAM_ONNX_ROS_CONFIG_HEADER} @ONLY)

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS ${SAM_ONNX_ROS_PUBLIC_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${SAM_ONNX_ROS_CATKIN_DEPENDS}
  DEPENDS ${${PROJECT_NAME}_CUDA_CATKIN_DEPENDS} OpenCV console_bridge
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

include_directories(
  include
  SYSTEM
  ${${PROJECT_NAME}_CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${console_bridge_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/sam_inference.cpp
  src/segmentation.cpp
  src/utils.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${console_bridge_LIBRARIES}
  ${${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES}
)

if(SAM_ONNX_ROS_TENSORRT_ENABLED)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sam_trt_lib/src/engineTRT.cpp")
    message(FATAL_ERROR "SAM_ONNX_ROS_TENSORRT_ENABLED is ON but the sam_trt_lib submodule is missing. Run: git submodule update --init --recursive")
  endif()
  target_sources(${PROJECT_NAME}
    PRIVATE
      3rdparty/sam_trt_lib/src/engineTRT.cpp
      3rdparty/sam_trt_lib/src/speedSam.cpp
  )
  target_include_directories(${PROJECT_NAME}
    PUBLIC
      3rdparty/sam_trt_lib/include
  )
endif()

# Main executable links the core lib but its used for testing (the lib is used on ed sensor integration)
add_executable(test_${PROJECT_NAME}
  src/main.cpp
)
target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${console_bridge_LIBRARIES})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM)

file(
  DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/speed_SAM/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM/SAM_encoder.onnx
  EXPECTED_HASH "SHA256=d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a"
  SHOW_PROGRESS
)
file(
  DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/speed_SAM/SAM_mask_decoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM/SAM_mask_decoder.onnx
  EXPECTED_HASH "SHA256=73b7c7bbe1ac78ada6f9534dfec5d4c31791fe8e57d552fa878c993020fae528"
  SHOW_PROGRESS
)


# ------------------------------------------------------------------------------------------------
#                                               INSTALL
# ------------------------------------------------------------------------------------------------

install(FILES
  ${SAM_ONNX_ROS_CONFIG_HEADER}
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  TARGETS
    ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
  TARGETS
    ${PROJECT_NAME}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY resources/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resources
)

# ------------------------------------------------------------------------------------------------
#                                           Testing
# ------------------------------------------------------------------------------------------------
if (CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test("-W2 --ignore EXTERNAL_DIRECTORY --ignore EXTERNAL_FILE --ignore EXTERNAL_INCLUDE_PATH --ignore HEADER_OUTSIDE_PACKAGE_INCLUDE_PATH --ignore MISSING_DIRECTORY")

  # Utils unit tests (no models needed)
  catkin_add_gtest(utils_tests test/test_utils.cpp)
  if(TARGET utils_tests)
    target_link_libraries(
      utils_tests
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      )
  endif()

  # SAM integration-ish tests (may need models)
  catkin_add_gtest(sam_tests test/sam_test.cpp WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM")
  if(TARGET sam_tests)
    target_link_libraries(
      sam_tests
      ${PROJECT_NAME}
      ${OpenCV_LIBRARIES}
      ${catkin_LIBRARIES}
    )
  endif()
endif()
