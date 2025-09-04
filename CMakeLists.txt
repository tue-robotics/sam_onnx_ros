cmake_minimum_required(VERSION 3.0.2)

project(sam_onnx_ros)

# -------------- CMake Policies ------------------#
#add_compile_options(-Wall -Werror=all)
#add_compile_options(-Wextra -Werror=extra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# -------------- ONNXRuntime  ------------------#
set(ONNXRUNTIME_VERSION 1.21.0)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../hero_sam/onnxruntime-linux-x64-gpu-1.21.1")
include_directories(${ONNXRUNTIME_ROOT}/include)

# -------------- Cuda ------------------#
add_definitions(-DUSE_CUDA=1)
include_directories(/usr/local/cuda/include)

find_package(catkin REQUIRED
  COMPONENTS
  # roscpp
  # tue_config
  # tue_filesystem
  # code_profiler

  #onnxruntime_ros
)

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS include
  #LIBRARIES ${PROJECT_NAME}
  LIBRARIES sam_onnx_ros_core
  CATKIN_DEPENDS
  DEPENDS OpenCV
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

include_directories(
  include
  SYSTEM
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

set(PROJECT_SOURCES
        src/main.cpp
        src/sam_inference.cpp
        src/segmentation.cpp
        src/utils.cpp
)

# Build core library (no main.cpp here)
add_library(sam_onnx_ros_core
        src/sam_inference.cpp
        src/segmentation.cpp
        src/utils.cpp
)
target_link_libraries(sam_onnx_ros_core
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)
target_include_directories(sam_onnx_ros_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main executable links the core lib
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} sam_onnx_ros_core)

# Copy sam_<model>.onnx file to the same folder of the executable file
configure_file(~/Documents/repos/hero_sam/sam_inference/model/SAM_mask_decoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_mask_decoder.onnx  COPYONLY)
configure_file(~/Documents/repos/hero_sam/sam_inference/model/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_encoder.onnx COPYONLY)

# Create folder name images in the same folder of the executable file
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
)

# # Enable testing
if (CATKIN_ENABLE_TESTING)
#   find_package(catkin_lint_cmake REQUIRED)
#   catkin_add_catkin_lint_test("-W2 --ignore HEADER_OUTSIDE_PACKAGE_INCLUDE_PATH")

  catkin_add_gtest(sam_onnx_ros_tests test/sam_test.cpp)
  if(TARGET sam_onnx_ros_tests)
    target_link_libraries(sam_onnx_ros_tests sam_onnx_ros_core ${catkin_LIBRARIES})
    target_include_directories(sam_onnx_ros_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
endif()

#If you want to debug
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
