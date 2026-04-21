#include "engineTRT.h"
#include "logging.h"
#include "cuda_utils.h"
#include "config.h"
#include "macros.h"
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <NvOnnxParser.h>
#include <opencv2/dnn.hpp>
#include <cuda_runtime_api.h>

static Logger gLogger;

std::string getFileExtension(const std::string& filePath) {
    // Find the position of the last dot in the file path
    size_t dotPos = filePath.find_last_of(".");
    // If a dot is found, extract and return the substring after the dot as the file extension
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos + 1);
    }
    // Return an empty string if no extension is found
    return "";
}

EngineTRT::EngineTRT(string modelPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16) {
    mRuntime = nullptr;
    mEngine = nullptr;
    mContext = nullptr;
    mCudaStream = nullptr;
    mInputNames = std::move(inputNames);
    mOutputNames = std::move(outputNames);

    // Check if the model file has an ".onnx" extension
    if (getFileExtension(modelPath) == "onnx") {
        // If the file is an ONNX model, build the engine using the provided parameters
        cout << "Building Engine from " << modelPath << endl;
        build(modelPath, mInputNames, mOutputNames, isDynamicShape, isFP16);
    }
    else {
        // If the file is not an ONNX model, deserialize an existing engine
        cout << "Deserializing Engine." << endl;
        deserializeEngine(modelPath, mInputNames, mOutputNames);
    }
}

EngineTRT::~EngineTRT() {
    // Release the CUDA stream
    if (mCudaStream)
        cudaStreamDestroy(mCudaStream);
    // Free GPU buffers allocated for inference
    for (size_t i = 0; i < mGpuBuffers.size(); ++i)
        if (mGpuBuffers[i])
            CUDA_CHECK(cudaFree(mGpuBuffers[i]));
    // Free CPU buffers
    for (size_t i = 0; i < mCpuBuffers.size(); ++i)
        delete[] mCpuBuffers[i];

    // Clean up and destroy the TensorRT engine components
    delete mContext;  // Destroy the execution context
    delete mEngine;   // Destroy the engine
    delete mRuntime;  // Destroy the runtime
}

void EngineTRT::build(string onnxPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16)
{
    // Check if the ONNX file exists. If not, print an error message and return.
    if (!std::filesystem::exists(onnxPath)) {
        std::cerr << "ONNX file not found: " << onnxPath << std::endl;
        return;  // Early exit if the ONNX file is missing
    }

    // Create an inference builder for building the TensorRT engine.
    auto builder = createInferBuilder(gLogger);
    assert(builder != nullptr);  // Ensure the builder is created successfully

    // Use explicit batch size, which is needed for ONNX models.
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);  // Ensure the network is created successfully

    // Create a builder configuration object to set options like FP16 precision.
    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);  // Ensure the config is created successfully

    // If dynamic shape support is needed, configure the optimization profile.
    if (isDynamicShape) // Only designed for NanoSAM mask decoder
    {
        // Create an optimization profile for dynamic input shapes.
        auto profile = builder->createOptimizationProfile();

        // Set the minimum, optimal, and maximum dimensions for the first input.
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMIN, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kOPT, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMAX, Dims3{ 1, 10, 2 });

        // Set the minimum, optimal, and maximum dimensions for the second input.
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMIN, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kOPT, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMAX, Dims2{ 1, 10 });

        // Add the optimization profile to the builder configuration.
        config->addOptimizationProfile(profile);
    }

    // Enable FP16 mode if specified.
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);  // Use mixed precision for faster inference
    }

    // Create a parser to convert the ONNX model to a TensorRT network.
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    assert(parser != nullptr);  // Ensure the parser is created successfully

    // Parse the ONNX model from the specified file.
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        throw std::runtime_error("Failed to parse ONNX file: " + onnxPath);
    }

    // Serialize the built network into a binary plan for execution.
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    assert(plan != nullptr);  // Ensure the network was serialized successfully

    // Create a runtime object for deserializing the engine.
    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime != nullptr);  // Ensure the runtime is created successfully

    // Deserialize the serialized plan to create an execution engine.
    mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size());
    assert(mEngine != nullptr);  // Ensure the engine was deserialized successfully

    // Create an execution context for running inference.
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);  // Ensure the context is created successfully

    // Clean up resources.
    delete network;
    delete config;
    delete parser;
    delete plan;

    // Initialize the engine with the input and output names.
    initialize(inputNames, outputNames);
}

void EngineTRT::saveEngine(const std::string& engineFilePath) {
    if (mEngine) {
        // Serialize the engine to a binary format.
        IHostMemory* serializedEngine = mEngine->serialize();
        std::ofstream engineFile(engineFilePath, std::ios::binary);
        if (engineFile) {
            // Write the serialized engine data to the specified file.
            engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
            std::cout << "Serialized engine saved to " << engineFilePath << std::endl;
        }
        delete serializedEngine;  // Destroy the serialized engine memory
    }
}

void EngineTRT::deserializeEngine(string engine_name, vector<string> inputNames, vector<string> outputNames)
{
    // Open the engine file in binary mode.
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);  // Trigger an assertion failure if the file cannot be opened
    }

    // Determine the size of the file and read the serialized engine data.
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    assert(serializedEngine);  // Ensure memory allocation was successful
    file.read(serializedEngine, size);
    file.close();

    // Create a runtime object and deserialize the engine.
    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime);  // Ensure the runtime is created successfully
    mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);
    mContext = mEngine->createExecutionContext();
    delete[] serializedEngine;  // Free the serialized engine memory

    // Ensure the number of bindings matches the expected number of inputs and outputs.
    assert(static_cast<size_t>(mEngine->getNbIOTensors()) == inputNames.size() + outputNames.size());

    // Initialize the engine with the input and output names.
    initialize(inputNames, outputNames);
}

void EngineTRT::initialize(vector<string> inputNames, vector<string> outputNames)
{
    (void)inputNames;
    (void)outputNames;

    // Resize the GPU and CPU buffer vectors to accommodate all the engine bindings
    const int32_t nbTensors = mEngine->getNbIOTensors();
    mTensorNames.clear();
    mTensorModes.clear();
    mInputDims.clear();
    mOutputDims.clear();
    mGpuBuffers.resize(nbTensors, nullptr);
    mCpuBuffers.resize(nbTensors, nullptr);
    mSkipCpuMemcpy.resize(nbTensors, false);
    mBufferBindingBytes.clear();
    mBufferBindingSizes.clear();

    // Loop through all bindings to allocate memory and store dimension information
    for (int32_t i = 0; i < nbTensors; ++i)
    {
        const char* tensorName = mEngine->getIOTensorName(i);
        const Dims tensorDims = mEngine->getTensorShape(tensorName);
        const TensorIOMode tensorMode = mEngine->getTensorIOMode(tensorName);

        mTensorNames.emplace_back(tensorName);
        mTensorModes.push_back(tensorMode);

        // Calculate the size required for the binding based on its dimensions
        const size_t binding_size = getSizeByDim(tensorDims);
        mBufferBindingSizes.push_back(binding_size);  // Store the size of the binding
        mBufferBindingBytes.push_back(binding_size * sizeof(float));  // Calculate the size in bytes

        // Allocate host memory for the CPU buffer
        mCpuBuffers[i] = new float[binding_size];

        // Allocate device memory for the GPU buffer
        CUDA_CHECK(cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]));

        // Store input and output dimensions separately based on whether the binding is an input or output
        if (tensorMode == TensorIOMode::kINPUT)
        {
            mInputDims.push_back(tensorDims);
        }
        else
        {
            mOutputDims.push_back(tensorDims);
        }
    }

    // Create a CUDA stream for asynchronous operations
    CUDA_CHECK(cudaStreamCreate(&mCudaStream));

    for (size_t i = 0; i < mTensorNames.size(); ++i)
    {
        mContext->setTensorAddress(mTensorNames[i].c_str(), mGpuBuffers[i]);
    }
}

bool EngineTRT::infer()
{
    // Copy data from host (CPU) input buffers to device (GPU) input buffers asynchronously
    copyInputToDeviceAsync(mCudaStream);

    // Perform inference using TensorRT.
    bool status = mContext->enqueueV3(mCudaStream);

    if (!status)
    {
        // If inference fails, print an error message and return false
        cout << "inference error!" << endl;
        return false;
    }

    // Copy the results from device (GPU) output buffers to host (CPU) output buffers asynchronously
    copyOutputToHostAsync(mCudaStream);
    CUDA_CHECK(cudaStreamSynchronize(mCudaStream));

    // Return true if inference was successful
    return true;
}

void EngineTRT::copyInputToDeviceAsync(const cudaStream_t& stream)
{
    // Perform asynchronous memory copy from CPU to GPU for the input buffers
    memcpyBuffers(true, false, true, stream);
}

void EngineTRT::copyOutputToHostAsync(const cudaStream_t& stream)
{
    // Calls memcpyBuffers to handle the copying of data from GPU to CPU memory.
    // Arguments: false (do not copy input buffers), true (copy data from device to host),
    // true (perform the copy asynchronously), and the given CUDA stream.
    memcpyBuffers(false, true, true, stream);
}

void EngineTRT::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
    // Loop through all bindings (inputs and outputs) in the TensorRT engine.
    for (size_t i = 0; i < mTensorNames.size(); ++i)
    {
        // Skip memory copy for buffers that are externally bound (zero-copy PCIe optimization)
        if (mSkipCpuMemcpy[i]) continue;

        // Determine the destination and source pointers based on the copy direction.
        void* dstPtr = deviceToHost ? mCpuBuffers[i] : mGpuBuffers[i];
        const void* srcPtr = deviceToHost ? mGpuBuffers[i] : mCpuBuffers[i];
        // Get the size of the buffer in bytes.
        const size_t byteSize = mBufferBindingBytes[i];
        // Set the type of memory copy operation based on the direction.
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        // Check if the current binding is an input or output and copy accordingly.
        if ((copyInput && mTensorModes[i] == TensorIOMode::kINPUT)
            || (!copyInput && mTensorModes[i] == TensorIOMode::kOUTPUT))
        {
            if (async)
            {
                // Perform asynchronous memory copy using the CUDA stream.
                CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
            }
            else
            {
                // Perform synchronous memory copy.
                CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }
}

size_t EngineTRT::getSizeByDim(const Dims& dims)
{
    size_t size = 1;

    // Loop through each dimension and multiply to calculate the total size.
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        // If the dimension is -1 (dynamic), use a predefined maximum size.
        if (dims.d[i] == -1)
            size *= MAX_NUM_PROMPTS;
        else
            size *= dims.d[i];
    }

    return size;
}

void EngineTRT::setInput(Mat& image)
{
    // Fast, SIMD-vectorized OpenCV normalization and HWC->CHW planar transposition.
    // Scales by 1/255.0 and swaps BGR to RGB channel order.
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

    // Copy the continuous memory blob directly directly to the TensorRT input buffer
    memcpy(mCpuBuffers[0], blob.ptr<float>(), blob.total() * sizeof(float));
}

void EngineTRT::setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints)
{
    const size_t coordsIdx = getTensorIndex(mInputNames[1]);
    const size_t labelsIdx = getTensorIndex(mInputNames[2]);

    // We do NOT delete and malloc here anymore.
    // mCpuBuffers and mGpuBuffers were already allocated during build()->initialize()
    // to Dims3{1, 10, 2} which encompasses (MAX_NUM_PROMPTS * 2 * sizeof(float)).
    // Reallocating per frame causes massive PCIe stalling.

    // Calculate how many bytes we need specifically for THIS prompt size so we don't
    // copy zeroes or garbage data to the GPU unnecessarily.
    mBufferBindingBytes[coordsIdx] = sizeof(float) * numPoints * 2;
    mBufferBindingBytes[labelsIdx] = sizeof(float) * numPoints;

    // Copy input data into pre-allocated CPU buffers
    // We strictly skip copying 'features' because it's bound via Zero-Copy Device Pointers.
    if (features != nullptr && !mSkipCpuMemcpy[getTensorIndex(mInputNames[0])]) {
        memcpy(mCpuBuffers[getTensorIndex(mInputNames[0])], features, mBufferBindingBytes[getTensorIndex(mInputNames[0])]);
    }

    memcpy(mCpuBuffers[coordsIdx], imagePointCoords, sizeof(float) * numPoints * 2);
    memcpy(mCpuBuffers[labelsIdx], imagePointLabels, sizeof(float) * numPoints);
    memcpy(mCpuBuffers[getTensorIndex(mInputNames[3])], maskInput, mBufferBindingBytes[getTensorIndex(mInputNames[3])]);
    memcpy(mCpuBuffers[getTensorIndex(mInputNames[4])], hasMaskInput, mBufferBindingBytes[getTensorIndex(mInputNames[4])]);

    // Configure TensorRT to use a dynamic input shape targeting the exact subset of the buffer we populated
    mContext->setOptimizationProfileAsync(0, mCudaStream);
    mContext->setInputShape(mInputNames[1].c_str(), Dims3{ 1, numPoints, 2 });
    mContext->setInputShape(mInputNames[2].c_str(), Dims2{ 1, numPoints });
    mContext->setTensorAddress(mInputNames[1].c_str(), mGpuBuffers[coordsIdx]);
    mContext->setTensorAddress(mInputNames[2].c_str(), mGpuBuffers[labelsIdx]);
}

void EngineTRT::getOutput(float* features)
{
    // Copy the output features from the CPU buffer to the provided memory
    const size_t outputIdx = getTensorIndex(mOutputNames[0]);
    memcpy(features, mCpuBuffers[outputIdx], mBufferBindingBytes[outputIdx]);
}

void EngineTRT::getOutput(float* iouPrediction, float* lowResolutionMasks)
{
    // Copy the low-resolution masks and IOU predictions from the CPU buffers
    const size_t iouIdx = getTensorIndex(mOutputNames[0]);
    const size_t masksIdx = getTensorIndex(mOutputNames[1]);
    memcpy(iouPrediction, mCpuBuffers[iouIdx], mBufferBindingBytes[iouIdx]);
    memcpy(lowResolutionMasks, mCpuBuffers[masksIdx], mBufferBindingBytes[masksIdx]);
}

size_t EngineTRT::getTensorIndex(const std::string& tensorName) const
{
    for (size_t i = 0; i < mTensorNames.size(); ++i)
    {
        if (mTensorNames[i] == tensorName)
        {
            return i;
        }
    }

    throw std::runtime_error("Tensor not found in engine: " + tensorName);
}

void* EngineTRT::getDevicePtr(const std::string& tensorName) const
{
    return mGpuBuffers[getTensorIndex(tensorName)];
}

void EngineTRT::setDevicePtr(const std::string& tensorName, void* devicePtr)
{
    size_t idx = getTensorIndex(tensorName);

    // We do NOT free the existing mGpuBuffers[idx] if we own it, because it was allocated
    // at initialization. But to link it safely without memory leaks, we can just switch the context pointer.
    // However, we probably don't want to lose track of our original cudaMalloc if we ever restart.
    // For now, we will simply point TensorRT to the new external address.
    mContext->setTensorAddress(tensorName.c_str(), devicePtr);

    // Mark it so we completely skip PCIe copying over this buffer.
    mSkipCpuMemcpy[idx] = true;
}
