# Project Guidelines

## Code Style
- Follow ROS/C++14 standards.
- Use the Pimpl (Pointer to Implementation) idiom for heavy neural networks to avoid polluting headers (e.g., `SamSegPipeline::Impl`).

## Architecture
- **SAM & YOLO Integration**: Both models are dynamically bound and abstracted. `SamWrapper` encapsulates backends (`kOnnx`, `kSpeedSam`).
- **Memory Persistence**: Never allocate deep learning inference models per-frame. Models (`YOLO_V8` & `SpeedSam`) are instanced exactly once to prevent high-latency lockups (+60 seconds), and inference occurs separately in `SamSegPipeline::process()`.
- **Initialization Timing**: Do not block constructors with inference initialization. `SamSegPipeline::initialize()` uses lazy instantiation (called during the first `cluster` trigger segment phase) to avoid freezing ROS tf tree initializations.
- **Pre-SAM Filtering**: Aggressively filter YOLO outputs *before* passing them to SAM. Bounding boxes are explicitly discarded if they inherently lack 3D depth valid points natively projecting inside the point cloud frustum, or if their class structurally matches an `ignore_label` (e.g. the surface of the `"dining table"` itself).
- **Pre/Post-Processing Parity**: Ensure math parity across backends. TensorRT and ONNX implementations must yield identical masks (e.g., exact bounds checking, binary 0.5 thresholding, mathematical identical operations, `/255.0` flat normalization instead of ImageNet).

## Build and Test
- **Build**: Use `catkin build --this` or `tue-make <package>` or `tue-make --this`.
- **Dependencies**:
  - TensorRT requires hardcoded static paths in `CMakeLists.txt` and explicit `catkin_package()` includes targeting `sam_tensor_rt`.
  - Review [TENSORRT_COMPILE_DIFF_REPORT.md](../sam_tensor_rt/TENSORRT_COMPILE_DIFF_REPORT.md) and commit `129cc9d` for specific linking and compilation strategies that made `.engine` code build globally through `ed_sensor_integration`.
- **Testing (Next Steps)**: Validate the SAM TensorRT outputs identically match standard SAM ONNX outputs during active node iterations. Needs live integration tests.

## Conventions
- **Logging**: Use standard ROS macros `ROS_WARN`, `ROS_DEBUG`, etc. Always escape newlines `\n` safely to prevent compiler macro breaks.
