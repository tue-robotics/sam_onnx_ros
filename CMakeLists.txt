cmake_minimum_required(VERSION 3.5)

project(sam_onnx_ros)

add_compile_options(-Wall -Werror=all)
add_compile_options(-Wextra -Werror=extra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(console_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED
  COMPONENTS
  onnxruntime_ros
)

option(SAM_ONNX_ROS_ENABLE_TENSORRT "Build the optional SpeedSAM/TensorRT backend when TensorRT is available" ON)

set(${PROJECT_NAME}_CUDA_ENABLED ${onnxruntime_ros_CUDA_ENABLED})
if(${PROJECT_NAME}_CUDA_ENABLED)
  find_package(CUDAToolkit REQUIRED)
  set(${PROJECT_NAME}_CUDA_CATKIN_DEPENDS "CUDAToolkit")
  set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
  set(${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES "CUDA::cudart")
endif()

set(SAM_ONNX_ROS_TENSORRT_ENABLED FALSE)
set(SAM_ONNX_ROS_TENSORRT_INCLUDE_DIR "")
set(SAM_ONNX_ROS_TENSORRT_LIBRARIES "")
set(SAM_ONNX_ROS_TENSORRT_ROOT "/home/amigo/TensorRT-10.16.1.11")

if(SAM_ONNX_ROS_ENABLE_TENSORRT)
  if(NOT CUDAToolkit_FOUND)
    find_package(CUDAToolkit QUIET)
  endif()

  set(SAM_ONNX_ROS_TENSORRT_HINTS "/home/amigo/TensorRT-10.16.1.11")
  if(DEFINED ENV{TENSORRT_ROOT})
    set(SAM_ONNX_ROS_TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
    list(APPEND SAM_ONNX_ROS_TENSORRT_HINTS "$ENV{TENSORRT_ROOT}")
  endif()
  if(DEFINED TENSORRT_ROOT)
    set(SAM_ONNX_ROS_TENSORRT_ROOT "${TENSORRT_ROOT}")
    list(APPEND SAM_ONNX_ROS_TENSORRT_HINTS "${TENSORRT_ROOT}")
  endif()

  if(SAM_ONNX_ROS_TENSORRT_ROOT
     AND EXISTS "${SAM_ONNX_ROS_TENSORRT_ROOT}/include/NvInfer.h"
     AND EXISTS "${SAM_ONNX_ROS_TENSORRT_ROOT}/lib/libnvinfer.so"
     AND EXISTS "${SAM_ONNX_ROS_TENSORRT_ROOT}/lib/libnvonnxparser.so")
    set(SAM_ONNX_ROS_TENSORRT_INCLUDE_DIR "${SAM_ONNX_ROS_TENSORRT_ROOT}/include")
    set(SAM_ONNX_ROS_NVINFER_LIBRARY "${SAM_ONNX_ROS_TENSORRT_ROOT}/lib/libnvinfer.so")
    set(SAM_ONNX_ROS_NVONNXPARSER_LIBRARY "${SAM_ONNX_ROS_TENSORRT_ROOT}/lib/libnvonnxparser.so")
  endif()

  if(NOT SAM_ONNX_ROS_TENSORRT_INCLUDE_DIR)
    find_path(SAM_ONNX_ROS_TENSORRT_INCLUDE_DIR
      NAMES NvInfer.h
      HINTS ${SAM_ONNX_ROS_TENSORRT_HINTS}
      PATH_SUFFIXES include
    )
  endif()
  if(NOT SAM_ONNX_ROS_NVINFER_LIBRARY)
    find_library(SAM_ONNX_ROS_NVINFER_LIBRARY
      NAMES nvinfer
      HINTS ${SAM_ONNX_ROS_TENSORRT_HINTS}
      PATH_SUFFIXES lib lib64 lib/x64
    )
  endif()
  if(NOT SAM_ONNX_ROS_NVONNXPARSER_LIBRARY)
    find_library(SAM_ONNX_ROS_NVONNXPARSER_LIBRARY
      NAMES nvonnxparser
      HINTS ${SAM_ONNX_ROS_TENSORRT_HINTS}
      PATH_SUFFIXES lib lib64 lib/x64
    )
  endif()

  if(CUDAToolkit_FOUND AND SAM_ONNX_ROS_TENSORRT_INCLUDE_DIR AND SAM_ONNX_ROS_NVINFER_LIBRARY AND SAM_ONNX_ROS_NVONNXPARSER_LIBRARY)
    set(SAM_ONNX_ROS_TENSORRT_ENABLED TRUE)
    list(APPEND SAM_ONNX_ROS_TENSORRT_LIBRARIES
      ${SAM_ONNX_ROS_NVINFER_LIBRARY}
      ${SAM_ONNX_ROS_NVONNXPARSER_LIBRARY}
      CUDA::cudart
    )
    message(STATUS "TensorRT backend enabled.")
  else()
    message(STATUS "TensorRT backend disabled: TensorRT headers/libraries were not found. The ONNX backend remains available.")
  endif()
endif()

configure_file(include/${PROJECT_NAME}/config.hpp.in ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}/config.hpp)

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS include sam_tensor_rt/include ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS onnxruntime_ros
  DEPENDS ${${PROJECT_NAME}_CUDA_CATKIN_DEPENDS} OpenCV console_bridge
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

include_directories(
  include
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
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
  target_sources(${PROJECT_NAME}
    PRIVATE
      sam_tensor_rt/src/engineTRT.cpp
      sam_tensor_rt/src/speedSam.cpp
  )
  target_include_directories(${PROJECT_NAME}
    PUBLIC
      sam_tensor_rt/include
  )
  target_include_directories(${PROJECT_NAME}
    SYSTEM PRIVATE
      ${SAM_ONNX_ROS_TENSORRT_INCLUDE_DIR}
  )
  target_link_libraries(${PROJECT_NAME} ${SAM_ONNX_ROS_TENSORRT_LIBRARIES})
endif()

# Main executable links the core lib but its used for testing (the lib is used on ed sensor integration)
add_executable(test_${PROJECT_NAME}
  src/main.cpp
)
target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${console_bridge_LIBRARIES})

set(SAM_ONNX_ROS_RESOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/resources/speed_SAM)
set(SAM_ONNX_ROS_LOCAL_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sam_tensor_rt/model)
file(MAKE_DIRECTORY ${SAM_ONNX_ROS_RESOURCE_DIR})

if(EXISTS ${SAM_ONNX_ROS_LOCAL_MODEL_DIR}/SAM_encoder.onnx)
  file(COPY ${SAM_ONNX_ROS_LOCAL_MODEL_DIR}/SAM_encoder.onnx DESTINATION ${SAM_ONNX_ROS_RESOURCE_DIR})
else()
  file(
    DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/speed_SAM/SAM_encoder.onnx ${SAM_ONNX_ROS_RESOURCE_DIR}/SAM_encoder.onnx
    EXPECTED_HASH "SHA256=d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a"
    SHOW_PROGRESS
  )
endif()

if(EXISTS ${SAM_ONNX_ROS_LOCAL_MODEL_DIR}/SAM_mask_decoder.onnx)
  file(COPY ${SAM_ONNX_ROS_LOCAL_MODEL_DIR}/SAM_mask_decoder.onnx DESTINATION ${SAM_ONNX_ROS_RESOURCE_DIR})
else()
  file(
    DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/speed_SAM/SAM_mask_decoder.onnx ${SAM_ONNX_ROS_RESOURCE_DIR}/SAM_mask_decoder.onnx
    EXPECTED_HASH "SHA256=73b7c7bbe1ac78ada6f9534dfec5d4c31791fe8e57d552fa878c993020fae528"
    SHOW_PROGRESS
  )
endif()

# ------------------------------------------------------------------------------------------------
#                                               INSTALL
# ------------------------------------------------------------------------------------------------

install(FILES
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_INCLUDE_DESTINATION}/config.hpp
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
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resources
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resources
)

# ------------------------------------------------------------------------------------------------
#                                           Testing
# ------------------------------------------------------------------------------------------------
if (CATKIN_ENABLE_TESTING)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    message(WARNING "Skipping tests because catkin's bundled googletest is not compatible with CMake ${CMAKE_VERSION}.")
  else()
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
endif()
