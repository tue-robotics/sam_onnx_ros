cmake_minimum_required(VERSION 3.14)

project(sam_onnx_ros)

# -------------- CMake Policies ------------------#
# add_compile_options(-Wall -Werror=all)
# add_compile_options(-Wextra -Werror=extra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
find_package(console_bridge REQUIRED)

# -------------- ONNXRuntime  ------------------#
set(ONNXRUNTIME_VERSION 1.21.0)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../hero_sam.bak/onnxruntime-linux-x64-gpu-1.21.1")
include_directories(${ONNXRUNTIME_ROOT}/include)

# -------------- Cuda ------------------#
add_definitions(-DUSE_CUDA=1)
include_directories(/usr/local/cuda/include)

# -------------- Models ------------------#
# TODO: Find proper folder Copy sam_<model>.onnx file to the same folder of the executable file
configure_file(~/Documents/repos/hero_sam.bak/sam_inference/model/SAM_mask_decoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_mask_decoder.onnx  COPYONLY)
configure_file(~/Documents/repos/hero_sam.bak/sam_inference/model/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_encoder.onnx COPYONLY)


find_package(catkin REQUIRED
  COMPONENTS
  rosconsole
  console_bridge
  #onnxruntime_ros
)

find_package(console_bridge REQUIRED)

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_lib
  CATKIN_DEPENDS
  DEPENDS OpenCV console_bridge
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

# Build core library
add_library(${PROJECT_NAME}_lib
        src/sam_inference.cpp
        src/segmentation.cpp
        src/utils.cpp
)

target_include_directories(${PROJECT_NAME}_lib
  PUBLIC
    include
    SYSTEM
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${console_bridge_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)


target_link_libraries(${PROJECT_NAME}_lib
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# Main executable links the core lib
add_executable(${PROJECT_NAME}
  src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# ------------------------------------------------------------------------------------------------
#                                           Install Targets
# ------------------------------------------------------------------------------------------------

install(
  DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)

install(
  TARGETS
    ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
  TARGETS
    ${PROJECT_NAME}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ------------------------------------------------------------------------------------------------
#                                           Testing
# ------------------------------------------------------------------------------------------------
if (CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test("-W2 --ignore HEADER_OUTSIDE_PACKAGE_INCLUDE_PATH")

  # Utils unit tests (no models needed)
  catkin_add_gtest(utils_tests test/test_utils.cpp)
  if(TARGET utils_tests)
    target_link_libraries(
      utils_tests
      ${PROJECT_NAME}_lib
      ${catkin_LIBRARIES}
      GTest::gtest
      GTest::gtest_main
      )
    #target_include_directories(utils_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()

  # SAM integration-ish tests (may need models)
  catkin_add_gtest(sam_tests test/sam_test.cpp)
  if(TARGET sam_tests)
    target_link_libraries(
      sam_tests
      ${PROJECT_NAME}_lib
      ${catkin_LIBRARIES}
      GTest::gtest
      GTest::gtest_main
    )
    #target_include_directories(sam_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
endif()
