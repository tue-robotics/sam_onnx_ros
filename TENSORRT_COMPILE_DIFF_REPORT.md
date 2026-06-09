# TensorRT Compile Diff Report

Comparison baseline: commit `ab6ee9e98333c4a9f462592e521617f984a1e899`  
Compared against: current workspace based on `a2c10ea` plus local uncommitted changes

## Executive Summary

The TensorRT-related work is concentrated in `sam_tensor_rt/include/engineTRT.h` and `sam_tensor_rt/src/engineTRT.cpp`.
The main purpose of these edits was to move the code from the older binding-index TensorRT API to the newer tensor-name API that current TensorRT versions expect.

The most important API migrations are:

- `getNbBindings()` -> `getNbIOTensors()`
- `getBindingDimensions()` -> `getTensorShape()`
- `bindingIsInput()` -> `getTensorIOMode()`
- `executeV2()` -> `enqueueV3()`
- `setBindingDimensions()` -> `setInputShape()`
- explicit binding pointer passing -> `setTensorAddress()`
- old 3-argument `deserializeCudaEngine(...)` call -> current overload without the plugin factory argument

These changes are exactly the kind of updates needed when older TensorRT 8-era code is compiled against newer TensorRT headers.

## Build Verification

The current workspace builds successfully with:

```bash
cmake --build build --target test_sam_onnx_ros -j2
```

Result: `test_sam_onnx_ros` built successfully.

## File-By-File Changes

### 1. `sam_tensor_rt/include/engineTRT.h`

Changes:

- Added `<string>` and `<vector>` includes.
- Added `getTensorIndex(const std::string& tensorName) const`.
- Added persistent members for tensor bookkeeping:
  - `mInputNames`
  - `mOutputNames`
  - `mTensorNames`
  - `mTensorModes`

Why this was done:

- The old implementation assumed TensorRT I/O could be handled only by fixed binding indices.
- The new TensorRT execution flow is tensor-name based, so the code now needs to remember the declared input/output names and map them to the engine's actual I/O tensor ordering.
- `getTensorIndex()` is the helper that lets the runtime find the correct buffer slot for a named tensor.

Performance impact:

- No measurable inference slowdown from the new members themselves.
- `getTensorIndex()` does a short linear lookup over a very small number of tensors, so the runtime cost is negligible in this project.
- If this ever grows to many tensors, a hash map could remove that lookup cost, but that is not necessary here.

### 2. `sam_tensor_rt/src/engineTRT.cpp`

This file contains the real TensorRT compatibility work.

#### 2.1 Constructor and destructor hardening

Changes:

- Initializes `mRuntime`, `mEngine`, `mContext`, and `mCudaStream` to `nullptr`.
- Stores `inputNames` and `outputNames` into member state.
- Destructor now checks pointers before destroying/freeing them.
- Destructor now frees GPU buffers only when they are non-null.
- Loop indices were changed from `int` to `size_t` where appropriate.

Why this was done:

- Once the code started reallocating buffers dynamically and using newer APIs, the object needed safer ownership handling.
- The previous destructor could destroy/free uninitialized pointers if construction failed early.
- Saving tensor names in the object is required for the newer tensor-name execution flow.

Performance impact:

- No inference performance loss.
- This is safety and correctness work only.

#### 2.2 ONNX parsing and engine deserialization API updates

Changes:

- Replaced the unused `parsed` variable with an explicit failure check and exception.
- Removed `assert(mCudaStream != nullptr)` during build.
- Changed:

```cpp
mRuntime->deserializeCudaEngine(plan->data(), plan->size(), nullptr);
```

to:

```cpp
mRuntime->deserializeCudaEngine(plan->data(), plan->size());
```

- Changed `serializedEngine->destroy()` to `delete serializedEngine`.

Why this was done:

- The old `deserializeCudaEngine(..., nullptr)` form is not the API expected by newer TensorRT headers.
- `IHostMemory` cleanup style also changed in modern TensorRT C++ usage.
- The old `assert(mCudaStream != nullptr)` was incorrect at build time because the CUDA stream is only created later in `initialize()`.
- Throwing on parser failure is more useful than silently continuing with a bad parse.

Performance impact:

- No inference performance loss.
- This is compile compatibility and cleanup correctness.

#### 2.3 Binding API migration to I/O tensor API

Changes:

- Replaced `getNbBindings()` with `getNbIOTensors()`.
- Replaced `getBindingDimensions(i)` with `getTensorShape(tensorName)`.
- Replaced `bindingIsInput(i)` with `getTensorIOMode(tensorName)`.
- Replaced the old dummy `getBindingIndex(...)` loops with real tensor-name discovery.
- Stores engine tensor names and tensor modes during initialization.
- Calls `mContext->setTensorAddress(...)` for every tensor buffer after allocation.

Why this was done:

- This is the core TensorRT migration.
- Older TensorRT code treated inputs and outputs as numbered bindings.
- Newer TensorRT versions expose engine I/O as named tensors and expect addresses to be registered per tensor name.
- Without this migration, the code either does not compile or compiles but cannot execute correctly with `enqueueV3()`.

Performance impact:

- Startup cost is very slightly higher because initialization now records tensor names and modes.
- Inference throughput is effectively unchanged because buffer registration happens once during setup, not on every run.
- This should not reduce model execution performance.

#### 2.4 Inference path migration from `executeV2` to `enqueueV3`

Changes:

- Replaced:

```cpp
mContext->executeV2(mGpuBuffers.data());
```

with:

```cpp
mContext->enqueueV3(mCudaStream);
```

- Added `cudaStreamSynchronize(mCudaStream)` after copying outputs back to host.

Why this was done:

- `enqueueV3()` is the newer execution entry point used with tensor-address registration.
- Once execution became fully stream-based, the code needed an explicit synchronization point before returning, otherwise host code could read output buffers before the async GPU work had actually finished.
- The old version copied outputs asynchronously and returned immediately, which could be unsafe depending on when the caller consumed the data.

Performance impact:

- Model kernel performance is not reduced by moving to `enqueueV3()`.
- The added `cudaStreamSynchronize()` may remove some theoretical overlap between inference and later host-side work, but it also guarantees correctness.
- In practice this is best described as a correctness fix, not a TensorRT slowdown.
- If you later want maximum overlap, the API can still support it, but the caller would need to manage stream synchronization explicitly.

#### 2.5 Dynamic-shape input handling migration

Changes:

- The prompt-coordinate and prompt-label buffers are now found by tensor name instead of assuming they are always indices `1` and `2`.
- Old prompt buffers are freed before reallocating new ones.
- Replaced `setBindingDimensions(...)` with `setInputShape(...)`.
- Re-registers the resized prompt buffers with `setTensorAddress(...)`.

Why this was done:

- `setBindingDimensions(...)` belongs to the old binding-based API.
- With the new tensor-name flow, dynamic input sizes must be set by input tensor name.
- Re-registering the tensor addresses is required because the prompt buffers are reallocated when `numPoints` changes.
- Freeing the previous GPU prompt buffers avoids leaking device memory across repeated predictions.

Performance impact:

- No meaningful TensorRT execution slowdown from the API switch itself.
- The code still reallocates prompt buffers per call when the number of points changes, so there is allocator overhead, but that overhead already existed conceptually in the old implementation and is now safer because old GPU buffers are freed.
- If you want to optimize further, the best next step would be reusing preallocated max-size prompt buffers rather than reallocating every time.

#### 2.6 Output and buffer access changed from fixed indices to named lookup

Changes:

- `getOutput(float* features)` now copies from the tensor named by `mOutputNames[0]`.
- `getOutput(float* iouPrediction, float* lowResolutionMasks)` now resolves both output tensors by name.
- Input copying for decoder tensors also resolves by tensor name instead of hardcoded indices like `0`, `3`, `4`, `5`, `6`.

Why this was done:

- Hardcoded binding positions are fragile once TensorRT stops exposing the old binding-index API in the same way.
- Named lookup makes the code robust even if engine tensor ordering differs from the older assumptions.

Performance impact:

- The lookup cost is negligible because the number of tensors is tiny.
- No meaningful inference slowdown.

#### 2.7 Small cleanup changes

Changes:

- Removed unused local variables `inputH` and `inputW`.
- Changed the `Dims` loop index type in `getSizeByDim()` to `int32_t`.
- Wrapped some CUDA calls in `CUDA_CHECK(...)` that were previously unchecked.

Why this was done:

- These changes help compilation with stricter warnings and align types with TensorRT's API.
- They also make runtime failures easier to catch.

Performance impact:

- No meaningful performance impact.

### 3. `sam_tensor_rt/include/utils.h`

Changes:

- Added `(void)flags;` and `(void)param;` in `mouseCallback(...)`.
- Added `(void)flags;` in `onMouse(...)`.

Why this was done:

- This suppresses unused-parameter warnings.
- The project uses strict warnings as errors, so these tiny edits can be required to keep the build green on stricter compilers/settings.

Performance impact:

- None.

Compilation relevance:

- Helpful for warning-clean builds, but not part of the TensorRT API migration itself.

### 4. `sam_tensor_rt/assets/blue-linkedin-logo_speedsam_bbox_mask.png`
### 5. `sam_tensor_rt/assets/dog_speedsam_bbox_mask.png`
### 6. `sam_tensor_rt/assets/dog_speedsam_bbox_mask_speedsam_bbox_mask.png`
### 7. `sam_tensor_rt/assets/dogs_speedsam_bbox_mask.png`

Changes:

- Four new binary image files were added.

Why this was done:

- These look like generated segmentation output examples or demo artifacts.
- They are not referenced in the TensorRT build logic and are not required for compiling the TensorRT backend.

Performance impact:

- None at runtime unless they are loaded manually for demos.
- No compile impact.

### 8. `.vscode/settings.json`

Changes:

- Added `"ROS2.distro": "noetic"`.

Why this was done:

- This is an editor configuration change only.
- It is unrelated to TensorRT compilation and does not affect the built code.

Performance impact:

- None.

## What Actually Changed To Make TensorRT Compile

The changes that matter for TensorRT compilation are:

- `sam_tensor_rt/include/engineTRT.h`
- `sam_tensor_rt/src/engineTRT.cpp`

The specific compile-enabling reasons are:

- The code was migrated from deprecated/older binding-index TensorRT APIs to current tensor-name APIs.
- The engine execution path was updated to the `enqueueV3()` model.
- Dynamic input shape handling was updated from binding-based shape assignment to named tensor shape assignment.
- Engine/tensor buffer registration now uses `setTensorAddress(...)`, which matches the newer execution API.
- Older deserialization and memory-management call patterns were updated to signatures accepted by the current TensorRT SDK.

`sam_tensor_rt/include/utils.h` only helps keep warning-clean builds under strict compiler settings.
The image assets and VS Code settings are not part of the compile fix.

## Performance Conclusion

Overall conclusion:

- There is no clear model-performance regression caused by this TensorRT API migration.
- Most changes are API-compatibility and correctness work.
- The only place where throughput could look slightly worse is the new `cudaStreamSynchronize()` inside `infer()`, because it forces the call to behave synchronously before returning.
- That said, the old code was returning after async output copies without waiting, so the new behavior is safer and more correct.
- The bigger remaining performance opportunity is not the migration itself; it is the repeated prompt-buffer reallocation in `setInput(...)` for dynamic prompts.

## Recommended One-Line Summary

The TensorRT compile fix was mainly a migration from old binding-based TensorRT APIs to the newer tensor-name and `enqueueV3()` execution APIs, with no meaningful expected inference slowdown beyond a small correctness-driven synchronization cost and some existing prompt-buffer allocation overhead.
