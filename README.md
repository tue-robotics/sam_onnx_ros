# SAM C++ ONNX implementation

Inspired by SAM NN from meta and Tensor-RT implementation from: https://github.com/hamdiboukamcha/SPEED-SAM-C-TENSORRT.git

## 🌐 Overview
A high-performance C++ implementation for SAM (segment anything model) using ONNX and CUDA, optimized for real-time image segmentation tasks.


## 📢 Performance

### Warm-Up cost :fire:
    NVIDIA GeForce RTX 3050
    Encoder Cuda warm-up cost 66.875 ms.
    Decoder Cuda warm-up cost 53.87 ms.

 ### Infernce Time

| Component                  | Pre processing | Inference | Post processing |
|----------------------------|----------------| --------- | ----------------|
| **Image Encoder**          |           | ||
| Parameters                  | 5M        |- | -|
| Speed                       | 8ms       | 33.322ms | 0.437ms |
| **Mask Decoder**           |           | ||
| Parameters                  | 3.876M    |- |- |
| Speed                       | 34ms       | 11.176ms | 5.984|
| **Whole Pipeline (Enc+Dec)** |         | | |
| Parameters                  | 9.66M     | -| -|
| Sum of Speed                       | 92.92ms      | - |-  |


## 📂 Project Structure
    SPEED-SAM-CPP-TENSORRT/
    ├── include
    │   ├── config.h          # Model configuration and macros
    │   ├── cuda_utils.h      # CUDA utility macros
    │   ├── engineTRT.h       # TensorRT engine management
    │   ├── logging.h         # Logging utilities
    │   ├── macros.h          # API export/import macros
    │   ├── speedSam.h        # SpeedSam class definition
    │   └── utils.h           # Utility functions for image handling
    ├── src
    │   ├── engineTRT.cpp     # Implementation of the TensorRT engine
    │   ├── main.cpp          # Main entry point
    │   └── speedSam.cpp      # Implementation of the SpeedSam class
    └── CMakeLists.txt        # CMake configuration

# 🚀 Installation
## Compile
    git clone <repo>
    cd sam_onnx_ros
    # Create a build directory and compile
    mkdir build && cd build
    cmake ..
    make -j$(nproc)

Note: Update the CMakeLists.txt with the correct paths for Onnxruntime and OpenCV and Onnx Models (since for TechUnited we keep them on separate repositories).

You can use main.cpp to run the application

## ROS option
    You can also run the code as a catkin package.

## 📦 Dependencies
    CUDA: NVIDIA's parallel computing platform
    Onnx: High-performance deep learning inference
    OpenCV: Image processing library
    C++17: Required standard for compilation
