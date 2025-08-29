cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME sam_onnx_ros)

project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

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
  #onnxruntime_ros
)

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
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

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link OpenCV libraries along with ONNX Runtime
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)


# Copy sam_<model>.onnx file to the same folder of the executable file
configure_file(../hero_sam/sam_inference/model/SAM_mask_decoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_mask_decoder.onnx  COPYONLY)
configure_file(../hero_sam/sam_inference/model/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_encoder.onnx COPYONLY)

# Create folder name images in the same folder of the executable file
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
)

#If you want to debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
