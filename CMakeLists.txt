cmake_minimum_required(VERSION 3.5)

project(sam_onnx_ros)

# -------------- CMake Policies ------------------#
add_compile_options(-Wall -Werror=all)
add_compile_options(-Wextra -Werror=extra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# -------------- Models ------------------#
set(SAM_MODELS_DIR "$ENV{HOME}/Documents/repos/hero_sam.bak/sam_inference/model" CACHE PATH "SAM models dir")
if (EXISTS "${SAM_MODELS_DIR}/SAM_mask_decoder.onnx")
  configure_file(${SAM_MODELS_DIR}/SAM_mask_decoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_mask_decoder.onnx COPYONLY)
endif()
if (EXISTS "${SAM_MODELS_DIR}/SAM_encoder.onnx")
  configure_file(${SAM_MODELS_DIR}/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_encoder.onnx COPYONLY)
endif()

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

find_package(catkin REQUIRED
  COMPONENTS
  rosconsole
  onnxruntime_ros
)
find_package(console_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# -------------- CUDA (optional, version-safe) ------------------#
set(${PROJECT_NAME}_CUDA_ENABLED ${onnxruntime_ros_CUDA_ENABLED})
set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "")
set(${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES "")

if(${PROJECT_NAME}_CUDA_ENABLED)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
      list(APPEND ${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES CUDA::cudart)
    else()
      message(STATUS "CUDAToolkit not found; continuing without explicit cudart link")
    endif()
  else()
    message(STATUS "CMake ${CMAKE_VERSION} < 3.17: skipping find_package(CUDAToolkit); trying /usr/local/cuda")
    list(APPEND ${PROJECT_NAME}_CUDA_INCLUDE_DIRS /usr/local/cuda/include)
    find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
    if(CUDART_LIB)
      list(APPEND ${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES ${CUDART_LIB})
    endif()
  endif()
endif()


configure_file(include/${PROJECT_NAME}/config.hpp.in ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}/config.hpp)
# add_custom_target(generate_config_hpp
#   DEPENDS ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}/config.hpp

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS rosconsole onnxruntime_ros
  DEPENDS OpenCV console_bridge
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

# Build library
add_library(${PROJECT_NAME}
  src/sam_inference.cpp
  src/segmentation.cpp
  src/utils.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    include
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
    SYSTEM
    ${${PROJECT_NAME}_CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES}
)

# Main executable links the core lib but its used for testing (the lib is used on ed sensor integration)
add_executable(test_${PROJECT_NAME}
  src/main.cpp
)

target_link_libraries(test_${PROJECT_NAME}
  ${PROJECT_NAME}
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
      ${OpenCV_LIBRARIES}
      ${catkin_LIBRARIES}
      GTest::gtest
      GTest::gtest_main
    )
    #target_include_directories(sam_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
endif()
