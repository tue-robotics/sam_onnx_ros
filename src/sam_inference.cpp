#include "sam_onnx_ros/sam_inference.hpp"
#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/dl_types.hpp"
#include "sam_onnx_ros/utils.hpp"

#include <algorithm>
#include <console_bridge/console.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <exception>
#include <memory>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <regex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define benchmark

SAM::SAM() = default;

SAM::~SAM()
{
    // Clean up input/output node names
    for (auto& name : inputNodeNames_)
    {
        delete[] name;
    }
    for (auto& name : outputNodeNames_)
    {
        delete[] name;
    }
}

#if defined(SAM_ONNX_ROS_CUDA_ENABLED) && SAM_ONNX_ROS_CUDA_ENABLED
namespace Ort
{
template <> struct TypeToTensorType<half>
{
    static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};
} // namespace Ort
#endif

const char* SAM::createSession(SEG::DlInitParam& iParams)
{
    const char* ret = RET_OK;
    if (session_)
    {
        session_.reset(); // Release previous session_

        // Clear node names
        for (auto& name : inputNodeNames_)
        {
            delete[] name;
        }
        inputNodeNames_.clear();

        for (auto& name : outputNodeNames_)
        {
            delete[] name;
        }
        outputNodeNames_.clear();
    }

    std::regex const pattern("[\u4e00-\u9fa5]");
    bool const result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        ret = "[SAM]: Your model path is error. Change your model path without chinese characters.";
        CONSOLE_BRIDGE_logWarn("%s", ret);
        return ret;
    }

    try
    {
        imgSize_ = iParams.imgSize;
        modelType_ = iParams.modelType;
        cudaEnable_ = iParams.cudaEnable;
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Sam");
        Ort::SessionOptions session_option;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cuda_option;
            cuda_option.device_id = 0;
            try
            {
                session_option.AppendExecutionProvider_CUDA(cuda_option);
            }
            catch (const std::exception& e)
            {
                // E.g. no GPU or missing CUDA/cuDNN libraries at runtime
                CONSOLE_BRIDGE_logWarn("[SAM]: Failed to enable CUDA, falling back to CPU: %s", e.what());
                cudaEnable_ = false;
            }
        }

        session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_option.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        session_option.SetLogSeverityLevel(iParams.logSeverityLevel);

        const char* model_path = iParams.modelPath.c_str();

        session_ = std::make_unique<Ort::Session>(env_, model_path, session_option);
        Ort::AllocatorWithDefaultOptions const allocator;
        size_t const input_nodes_num = session_->GetInputCount();
        for (size_t i = 0; i < input_nodes_num; ++i)
        {
            Ort::AllocatedStringPtr const input_node_name = session_->GetInputNameAllocated(i, allocator);
            // Copy the whole name, a fixed-size buffer overflows on long node names
            const size_t name_size = std::strlen(input_node_name.get()) + 1;
            char* temp_buf = new char[name_size];
            std::copy_n(input_node_name.get(), name_size, temp_buf);
            inputNodeNames_.push_back(temp_buf);
        }

        size_t const output_nodes_num = session_->GetOutputCount();
        for (size_t i = 0; i < output_nodes_num; ++i)
        {
            Ort::AllocatedStringPtr const output_node_name = session_->GetOutputNameAllocated(i, allocator);
            const size_t name_size = std::strlen(output_node_name.get()) + 1;
            char* temp_buf = new char[name_size];
            std::copy_n(output_node_name.get(), name_size, temp_buf);
            outputNodeNames_.push_back(temp_buf);
        }

        options_ = Ort::RunOptions{nullptr};

        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        // auto output_type = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();

        warmUpSession(modelType_);
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        std::string const error_msg = "[SAM]: Failed to create session: " + std::string(e.what());
        CONSOLE_BRIDGE_logWarn("%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
}

const char* SAM::runSession(const cv::Mat& iImg,
                            std::vector<SEG::DlResult>& oResult,
                            SEG::ModelType modelType,
                            SEG::DlResult& result)
{
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif
    Utils utilities;
    const char* ret = RET_OK;
    cv::Mat processed_img;
    utilities.preProcess(iImg, imgSize_, processed_img);

    auto* blob = new float[processed_img.total() * 3];
    utilities.blobFromImage(processed_img, blob);
    std::vector<int64_t> input_node_dims;
    if (modelType == SEG::ModelType::SAM_SEGMENT_ENCODER)
    {
        // NCHW with H=imgSize[1], W=imgSize[0]
        input_node_dims = {1, 3, imgSize_.at(1), imgSize_.at(0)};
    }
    else if (modelType == SEG::ModelType::SAM_SEGMENT_DECODER)
    {
        input_node_dims = {1, 256, 64, 64};
    }

    tensorProcess(starttime_1, iImg, blob, input_node_dims, modelType, oResult, utilities, result);

    // ...existing code...
    return ret;
}

template <typename N>
const char* SAM::tensorProcess(clock_t& starttime_1,
                               const cv::Mat& iImg,
                               N& blob,
                               std::vector<int64_t>& inputNodeDims,
                               SEG::ModelType _modelType,
                               std::vector<SEG::DlResult>& oResult,
                               Utils& utilities,
                               SEG::DlResult& result)
{
    switch (_modelType)
    {
    case SEG::ModelType::SAM_SEGMENT_ENCODER:
        // case OTHER_SAM_MODEL:
        {
            Ort::Value const input_tensor = Ort::Value::CreateTensor<std::remove_pointer_t<N>>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                blob,
                3 * imgSize_.at(0) * imgSize_.at(1),
                inputNodeDims.data(),
                inputNodeDims.size());

#ifdef benchmark
            clock_t const starttime_2 = clock();
#endif // benchmark

            auto output_tensor = session_->Run(
                options_, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(), outputNodeNames_.size());
#ifdef benchmark
            clock_t const starttime_3 = clock();
#endif // benchmark

            Ort::TypeInfo const type_info = output_tensor.front().GetTypeInfo();
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_node_dims = tensor_info.GetShape();
            auto output = output_tensor.front().GetTensorMutableData<std::remove_pointer_t<N>>();
            delete[] blob;

            int64_t const embedding_size =
                output_node_dims[1] * output_node_dims[2] * output_node_dims[3]; // Flattened size
            result.embeddings.assign(output, output + embedding_size); // Save embeddings

#ifdef benchmark
            clock_t const starttime_4 = clock();
            double const pre_process_time = static_cast<double>(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
            double const process_time = static_cast<double>(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
            double const post_process_time = static_cast<double>(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
            if (cudaEnable_)
            {
                CONSOLE_BRIDGE_logInform(
                    "[SAM_encoder(CUDA)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                    pre_process_time,
                    process_time,
                    post_process_time);
            }
            else
            {
                CONSOLE_BRIDGE_logInform(
                    "[SAM_encoder(CPU)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                    pre_process_time,
                    process_time,
                    post_process_time);
            }
#endif // benchmark
            break;
        }
    case SEG::ModelType::SAM_SEGMENT_DECODER:
    {
        // Use embeddings from the last result
        std::vector<float> embeddings = result.embeddings;
        // Create tensor for decoder
        std::vector<int64_t> decoder_input_dims = {1, 256, 64, 64}; // Adjust based on your decoder's requirements

#ifdef benchmark
        clock_t starttime_2 = 0;
        clock_t starttime_3 = 0;
#endif // benchmark

        for (const auto& box : result.boxes)
        {
            Ort::Value decoder_input_tensor =
                Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                                embeddings.data(), // Use the embeddings from the encoder
                                                embeddings.size(), // Total number of elements
                                                decoder_input_dims.data(),
                                                decoder_input_dims.size());

            // Use center of bounding box as foreground point
            // float centerX = box.x + box.width / 2.0;
            // float centerY = box.y + box.height / 2.0;

            // Convert bounding box to points
            std::vector<float> point_coords = {
                static_cast<float>(box.x),
                static_cast<float>(box.y), // Top-left
                static_cast<float>(box.x + box.width),
                static_cast<float>(box.y + box.height) // Bottom-right
            };

            std::vector<float> point_coords_scaled;

            std::vector<int64_t> const point_coords_dims = {1, 2, 2}; // 2 points, each with (x, y)

            // Labels for the points
            std::vector<float> point_labels = {2.0f, 3.0f}; // Box prompt labels
            std::vector<int64_t> const point_labels_dims = {1, 2};

            // Create dummy mask_input and has_mask_input
            std::vector<float> mask_input(static_cast<size_t>(256 * 256), 0.0F); // Fill with zeros
            std::vector<int64_t> const mask_input_dims = {1, 1, 256, 256};

            std::vector<float> has_mask_input = {0.0f}; // No mask provided
            std::vector<int64_t> const has_mask_input_dims = {1};

            utilities.scaleBboxPoints(iImg, imgSize_, point_coords, point_coords_scaled);

            std::vector<Ort::Value> input_tensors = Utils::prepareInputTensor(decoder_input_tensor,
                                                                              point_coords_scaled,
                                                                              point_coords_dims,
                                                                              point_labels,
                                                                              point_labels_dims,
                                                                              mask_input,
                                                                              mask_input_dims,
                                                                              has_mask_input,
                                                                              has_mask_input_dims);

#ifdef benchmark
            starttime_2 = clock();
#endif // benchmark
            auto output_tensors = session_->Run(options_,
                                                inputNodeNames_.data(),
                                                input_tensors.data(),
                                                input_tensors.size(),
                                                outputNodeNames_.data(),
                                                outputNodeNames_.size());

#ifdef benchmark
            starttime_3 = clock();
#endif // benchmark

            Utils::postProcess(output_tensors, iImg, imgSize_, result);
        }

        // Add the result to oResult
        oResult.push_back(result);

        delete[] blob;

#ifdef benchmark
        clock_t const starttime_4 = clock();
        double const pre_process_time = static_cast<double>(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double const process_time = static_cast<double>(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double const post_process_time = static_cast<double>(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            CONSOLE_BRIDGE_logInform("[SAM_decoder(CUDA)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                                     pre_process_time,
                                     process_time,
                                     post_process_time);
        }
        else
        {
            CONSOLE_BRIDGE_logInform("[SAM_decoder(CPU)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                                     pre_process_time,
                                     process_time,
                                     post_process_time);
        }
#endif // benchmark
        break;
    }
    default: CONSOLE_BRIDGE_logError("[SAM]: Unsupported model type.");
    }

    return RET_OK;
}

char* SAM::warmUpSession(SEG::ModelType modelType)
{
    clock_t const starttime_1 = clock();
    Utils utilities;
    cv::Mat const i_img = cv::Mat(cv::Size(imgSize_.at(0), imgSize_.at(1)), CV_8UC3);
    cv::Mat processed_img;
    utilities.preProcess(i_img, imgSize_, processed_img);
    auto* blob = new float[i_img.total() * 3];
    utilities.blobFromImage(processed_img, blob);
    // NCHW: H=imgSize[1], W=imgSize[0]
    std::vector<int64_t> sam_input_node_dims = {1, 3, imgSize_.at(1), imgSize_.at(0)};
    switch (modelType)
    {
    case SEG::ModelType::SAM_SEGMENT_ENCODER:
    {
        Ort::Value const input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            blob,
            3 * static_cast<size_t>(imgSize_.at(0)) * static_cast<size_t>(imgSize_.at(1)),
            sam_input_node_dims.data(),
            sam_input_node_dims.size());
        auto output_tensors = session_->Run(
            options_, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(), outputNodeNames_.size());
        delete[] blob;
        clock_t const starttime_4 = clock();
        double const post_process_time = static_cast<double>(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            CONSOLE_BRIDGE_logInform("[SAM(CUDA)]: Cuda warm-up cost %.2f ms.", post_process_time);
        }

        break;
    }

    case SEG::ModelType::SAM_SEGMENT_DECODER:
    {
        std::vector<int64_t> const input_node_dims = {1, 256, 64, 64}; // BUG: That was 236 instead of 256
        // Use embeddings from the last result
        std::vector<float> dummy_embeddings(static_cast<size_t>(256 * 64 * 64),
                                            1.0F); // Fill with zeros or any dummy values
        std::vector<int64_t> decoder_input_dims = {1, 256, 64, 64}; // Adjust based on your decoder's requirements

        // Create dummy point coordinates and labels
        std::vector<cv::Rect> const bounding_boxes = {
            cv::Rect(0, 0, 100, 100), // Example bounding box with (x, y, width, height)
            // cv::Rect(0, 0, 473, 359) // Another example bounding box
        };
        for (const auto& bbox : bounding_boxes)
        {
            Ort::Value decoder_input_tensor =
                Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                                dummy_embeddings.data(), // Use the embeddings from the encoder
                                                dummy_embeddings.size(), // Total number of elements
                                                decoder_input_dims.data(),
                                                decoder_input_dims.size());

            // Convert bounding box to points
            // Use center of bounding box as foreground point
            float const center_x = static_cast<float>(bbox.x) + (static_cast<float>(bbox.width) / 2.0F);
            float const center_y = static_cast<float>(bbox.y) + (static_cast<float>(bbox.height) / 2.0F);

            std::vector<float> point_coords = {center_x, center_y}; // Center point (foreground)

            std::vector<int64_t> const point_coords_dims = {1, 1, 2}; // 2 points, each with (x, y)

            std::vector<float> point_coords_scaled;

            utilities.scaleBboxPoints(i_img, imgSize_, point_coords, point_coords_scaled);

            // Labels for the points
            std::vector<float> point_labels = {1.0f}; // All points are foreground
            std::vector<int64_t> const point_labels_dims = {1, 1};

            // Create dummy mask_input and has_mask_input
            std::vector<float> mask_input(static_cast<size_t>(256 * 256), 0.0F); // Fill with zeros
            std::vector<int64_t> const mask_input_dims = {1, 1, 256, 256};
            std::vector<float> has_mask_input = {0.0f}; // No mask provided
            std::vector<int64_t> const has_mask_input_dims = {1};

            std::vector<Ort::Value> input_tensors = Utils::prepareInputTensor(decoder_input_tensor,
                                                                              point_coords_scaled,
                                                                              point_coords_dims,
                                                                              point_labels,
                                                                              point_labels_dims,
                                                                              mask_input,
                                                                              mask_input_dims,
                                                                              has_mask_input,
                                                                              has_mask_input_dims);

            auto output_tensors = session_->Run(options_,
                                                inputNodeNames_.data(),
                                                input_tensors.data(),
                                                input_tensors.size(),
                                                outputNodeNames_.data(),
                                                outputNodeNames_.size());
        }

        delete[] blob;
        clock_t const starttime_4 = clock();
        double const post_process_time = static_cast<double>(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            CONSOLE_BRIDGE_logInform("[SAM(CUDA)]: Cuda warm-up cost %.2f ms.", post_process_time);
        }

        break;
    }
    }

    return RET_OK;
}
