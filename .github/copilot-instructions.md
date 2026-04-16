# Project Guidelines

## Code Style
- Follow ROS/C++14 standards.
- Use the Pimpl (Pointer to Implementation) idiom for heavy neural networks to avoid polluting headers (e.g., `SamSegPipeline::Impl`).

## Architecture
- **SAM & YOLO Integration**: Both models are dynamically bound and abstracted. `SamWrapper` encapsulates backends (`kOnnx`, `kSpeedSam`).
- **Memory Persistence**: Never allocate deep learning inference models per-frame. Models (`YOLO_V8` & `SpeedSam`) are instanced exactly once to prevent high-latency lockups (+60 seconds), and inference occurs separately in `SamSegPipeline::process()`.
- **Initialization Timing**: Do not block constructors with inference initialization. `SamSegPipeline::initialize()` uses lazy instantiation (called during the first `cluster` trigger segment phase) to avoid freezing ROS tf tree initializations.
- **Pre-SAM Filtering**: Aggressively filter YOLO outputs *before* passing them to SAM. Bounding boxes are explicitly discarded if they inherently lack 3D depth valid points natively projecting inside the point cloud frustum, or if their class structurally matches an `ignore_label` (e.g. the surface of the `"dining table"` itself).
- **Pre/Post-Processing Math Parity**: Ensure algorithmic parity across backends. TensorRT implementations must yield identical masks to ONNX by upholding these mathematical constraints:
  - *Image Normalization*: Removed ImageNet distribution scaling `((pixel/255.0) - mean) / std` during RGB/BGR conversions. Models now exclusively use a flat `(pixel/255.0)` numerical baseline scalar.
  - *IoU Mask Selection*: TensorRT must iterate across all internal prediction outputs to extract the specific mask dimension `bestMaskIndex * HIDDEN_DIM * HIDDEN_DIM` possessing the highest parsed `IouPrediction`, bypassing arbitrary 0-index defaults.
  - *Binary Thresholding*: The CNN's returning logits must be binarized (`cv::compare` with `CMP_GT` at `0.5f`) mapping them strictly into hard edge configurations (`0` vs `255`).
  - *Morphological Denoising*: Structural interpolation smoothing is strictly enforced via dynamically sized elliptical OpenCV kernels `std::max(5, std::min(w, h)/100)` using `MORPH_CLOSE` followed seamlessly by `MORPH_OPEN` to match ONNX geometric boundary signatures.

## Build and Test
- **Build**: Use `catkin build --this` or `tue-make <package>` or `tue-make --this`.
- **Dependencies**:
  - TensorRT requires hardcoded static paths in `CMakeLists.txt` and explicit `catkin_package()` includes targeting `sam_tensor_rt`.
  - Review [TENSORRT_COMPILE_DIFF_REPORT.md](../sam_tensor_rt/TENSORRT_COMPILE_DIFF_REPORT.md) and commit `129cc9d` for specific linking and compilation strategies that made `.engine` code build globally through `ed_sensor_integration`.
- **Testing (Next Steps)**: Validate the SAM TensorRT outputs identically match standard SAM ONNX outputs during active node iterations. Needs live integration tests.

## Conventions
- **Logging**: Use standard ROS macros `ROS_WARN`, `ROS_DEBUG`, etc. Always escape newlines `\n` safely to prevent compiler macro breaks.
