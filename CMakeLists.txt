cmake_minimum_required(VERSION 3.17)

project(sam_onnx_ros)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------- Support C++17 for using filesystem  ------------------#
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Werror=all -Wextra -Werror=extra)
endif()

find_package(ament_cmake REQUIRED)
find_package(console_bridge REQUIRED)
find_package(onnxruntime_ros REQUIRED)
find_package(OpenCV REQUIRED)
# CUDA is optional, it is required by the TensorRT backend and by onnxruntime_ros when built with CUDA
find_package(CUDAToolkit QUIET)

option(${PROJECT_NAME}_ENABLE_TENSORRT
  "Build the optional SpeedSAM/TensorRT backend when TensorRT is available" ON)
if(DEFINED SAM_ONNX_ROS_ENABLE_TENSORRT)
  set(${PROJECT_NAME}_ENABLE_TENSORRT ${SAM_ONNX_ROS_ENABLE_TENSORRT})
endif()

if(${PROJECT_NAME}_ENABLE_TENSORRT AND CUDAToolkit_FOUND)
  set(${PROJECT_NAME}_TENSORRT_ENABLED TRUE)
  find_package(tensorrt_ros REQUIRED)
else()
  if(${PROJECT_NAME}_ENABLE_TENSORRT)
    message(STATUS "CUDA toolkit not found, building without the TensorRT backend")
  endif()
  set(${PROJECT_NAME}_TENSORRT_ENABLED FALSE)
endif()

if(onnxruntime_ros_CUDA_ENABLED OR ${PROJECT_NAME}_TENSORRT_ENABLED)
  set(${PROJECT_NAME}_CUDA_ENABLED TRUE)
else()
  set(${PROJECT_NAME}_CUDA_ENABLED FALSE)
endif()

if(${PROJECT_NAME}_CUDA_ENABLED AND NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "onnxruntime_ros is built with CUDA, but the CUDA toolkit could not be found")
endif()

set(${PROJECT_NAME}_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(${PROJECT_NAME}_CONFIG_HEADER ${${PROJECT_NAME}_GENERATED_INCLUDE_DIR}/${PROJECT_NAME}/config.hpp)
# @ONLY changes substitution behavior from: default: replace both ${VAR} and @VAR@ with @ONLY: replace only @VAR@
configure_file(include/${PROJECT_NAME}/config.hpp.in ${${PROJECT_NAME}_CONFIG_HEADER} @ONLY)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

add_library(${PROJECT_NAME} SHARED
  src/sam_inference.cpp
  src/segmentation.cpp
  src/utils.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${${PROJECT_NAME}_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
# The public headers include ONNX Runtime and OpenCV, and cuda_fp16.h when CUDA is enabled
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    onnxruntime_ros::onnxruntime_ros
    ${OpenCV_LIBRARIES}
  PRIVATE
    console_bridge::console_bridge
)
if(${PROJECT_NAME}_CUDA_ENABLED)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
endif()

if(${PROJECT_NAME}_TENSORRT_ENABLED)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sam_trt_lib/src/engineTRT.cpp")
    message(FATAL_ERROR "${PROJECT_NAME}_TENSORRT_ENABLED is ON but the sam_trt_lib submodule is missing. "
      "Run: git submodule update --init --recursive")
  endif()
  # Separate object library, so the 3rdparty code is kept out of compile_commands.json and isn't linted
  set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
  add_library(${PROJECT_NAME}_sam_trt_lib OBJECT
    3rdparty/sam_trt_lib/src/engineTRT.cpp
    3rdparty/sam_trt_lib/src/speedSam.cpp
  )
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set_target_properties(${PROJECT_NAME}_sam_trt_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${PROJECT_NAME}_sam_trt_lib PUBLIC 3rdparty/sam_trt_lib/include)
  target_link_libraries(${PROJECT_NAME}_sam_trt_lib PUBLIC tensorrt_ros::tensorrt_ros CUDA::cudart ${OpenCV_LIBRARIES})

  target_sources(${PROJECT_NAME} PRIVATE $<TARGET_OBJECTS:${PROJECT_NAME}_sam_trt_lib>)
  target_include_directories(${PROJECT_NAME} PRIVATE 3rdparty/sam_trt_lib/include)
  target_link_libraries(${PROJECT_NAME} PRIVATE tensorrt_ros::tensorrt_ros)
endif()

# Main executable links the core lib but its used for testing (the lib is used on ed sensor integration)
add_executable(test_${PROJECT_NAME}
  src/main.cpp
)
target_link_libraries(test_${PROJECT_NAME} PRIVATE ${PROJECT_NAME})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM)

# Authenticate the downloads when a GITHUB_TOKEN is available, to avoid rate limiting
# The token only authenticates the downloads and doesn't affect the build result
set(${PROJECT_NAME}_DOWNLOAD_ARGS SHOW_PROGRESS)
if(NOT "$ENV{GITHUB_TOKEN}" STREQUAL "")
  list(APPEND ${PROJECT_NAME}_DOWNLOAD_ARGS HTTPHEADER "Authorization: Bearer $ENV{GITHUB_TOKEN}")
endif()

file(
  DOWNLOAD
    https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/speed_SAM/SAM_encoder.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM/SAM_encoder.onnx
  EXPECTED_HASH "SHA256=d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a"
  ${${PROJECT_NAME}_DOWNLOAD_ARGS}
)
file(
  DOWNLOAD
    https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/speed_SAM/SAM_mask_decoder.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM/SAM_mask_decoder.onnx
  EXPECTED_HASH "SHA256=73b7c7bbe1ac78ada6f9534dfec5d4c31791fe8e57d552fa878c993020fae528"
  ${${PROJECT_NAME}_DOWNLOAD_ARGS}
)

# ------------------------------------------------------------------------------------------------
#                                               INSTALL
# ------------------------------------------------------------------------------------------------

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  PATTERN "*.in" EXCLUDE
)

install(
  FILES ${${PROJECT_NAME}_CONFIG_HEADER}
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS test_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resources/
  DESTINATION share/${PROJECT_NAME}/resources
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(onnxruntime_ros OpenCV)
if(${PROJECT_NAME}_CUDA_ENABLED)
  ament_export_dependencies(CUDAToolkit)
endif()

# ------------------------------------------------------------------------------------------------
#                                           Testing
# ------------------------------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_clang_tidy REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)
  find_package(tue_lint_config REQUIRED)

  # Only lint our own code, not the sam_trt_lib submodule
  ament_clang_format(CONFIG_FILE ${tue_lint_config_DIR}/../config/.clang-format --clang-format-version=21
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/test)
  ament_clang_tidy(CONFIG_FILE ${tue_lint_config_DIR}/../config/.clang-tidy --clang-tidy-version=21
    ${CMAKE_BINARY_DIR}/compile_commands.json)
  ament_lint_cmake(MAX_LINE_LENGTH 120)
  ament_xmllint(MAX_LINE_LENGTH 120)

  # Utils unit tests (no models needed)
  ament_add_gtest(utils_tests test/test_utils.cpp)
  if(TARGET utils_tests)
    target_link_libraries(utils_tests ${PROJECT_NAME})
  endif()

  # SAM integration-ish tests (may need models)
  ament_add_gtest(sam_tests test/sam_test.cpp
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM")
  if(TARGET sam_tests)
    target_link_libraries(sam_tests ${PROJECT_NAME})
  endif()
endif()

ament_package()
