#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/sam_inference.hpp"
#include "sam_onnx_ros/utils.hpp"

#include <console_bridge/console.h>

#include <regex>

#define benchmark
// #define ROI

SAM::SAM()
{
}

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

const char* SAM::CreateSession(SEG::DL_INIT_PARAM& iParams)
{
    const char* Ret = RET_OK;
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

    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        Ret = "[SAM]:Your model path is error. Change your model path without chinese characters.";
        CONSOLE_BRIDGE_logWarn("%s", Ret);
        return Ret;
    }

    try
    {
        imgSize_ = iParams.imgSize;
        modelType_ = iParams.modelType;
        cudaEnable_ = iParams.cudaEnable;
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Sam");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }

        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

        const char* modelPath = iParams.modelPath.c_str();

        session_ = std::make_unique<Ort::Session>(env_, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session_->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; ++i)
        {
            Ort::AllocatedStringPtr input_node_name = session_->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames_.push_back(temp_buf);
        }
    
        size_t OutputNodesNum = session_->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; ++i)
        {
            Ort::AllocatedStringPtr output_node_name = session_->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames_.push_back(temp_buf);
        }

        options_ = Ort::RunOptions{nullptr};

        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        // auto output_type = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();

        WarmUpSession_(modelType_);
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[SAM]: ";
        const char* str2 = e.what();
        std::string str_result = std::string(str1) + std::string(str2);
        char* merged = new char[str_result.length() + 1];
        std::strcpy(merged, str_result.c_str());
        CONSOLE_BRIDGE_logWarn("%s", merged);
        delete[] merged;
        return "[SAM]: CreateSession failed.";
    }
}

const char* SAM::RunSession(const cv::Mat& iImg, std::vector<SEG::DL_RESULT>& oResult, SEG::MODEL_TYPE modelType, SEG::DL_RESULT& result)
{
    #ifdef benchmark
    clock_t starttime_1 = clock();
    #endif
    Utils utilities;
    const char* Ret = RET_OK;
    cv::Mat processedImg;
    utilities.PreProcess(iImg, imgSize_, processedImg);

    if (modelType < 4)
    {
        float* blob = new float[processedImg.total() * 3];
        utilities.BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims;
        if (modelType == SEG::SAM_SEGMENT_ENCODER)
        {
            // NCHW with H=imgSize[1], W=imgSize[0]  // FIX
            inputNodeDims = { 1, 3, imgSize_.at(1), imgSize_.at(0) }; // FIX
        }
        else if (modelType == SEG::SAM_SEGMENT_DECODER)
        {
            inputNodeDims = { 1, 256, 64, 64 };
        }

        TensorProcess_(starttime_1, iImg, blob, inputNodeDims, modelType, oResult, utilities, result);
    }

    // ...existing code...
    return Ret;
}

template <typename N>
const char* SAM::TensorProcess_(clock_t& starttime_1, const cv::Mat& iImg,
                               N& blob, std::vector<int64_t>& inputNodeDims,
                               SEG::MODEL_TYPE _modelType,
                               std::vector<SEG::DL_RESULT>& oResult,
                               Utils& utilities, SEG::DL_RESULT& result)
{
    switch (_modelType)
    {
        case SEG::SAM_SEGMENT_ENCODER:
        // case OTHER_SAM_MODEL:
        {
            Ort::Value inputTensor =
                Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
                    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                    blob, 3 * imgSize_.at(0) * imgSize_.at(1), inputNodeDims.data(),
                    inputNodeDims.size());

            #ifdef benchmark
            clock_t starttime_2 = clock();
            #endif // benchmark
            
            auto outputTensor = 
                session_->Run(options_, inputNodeNames_.data(), &inputTensor, 1,
                       outputNodeNames_.data(), outputNodeNames_.size());
            #ifdef benchmark
            clock_t starttime_3 = clock();
            #endif // benchmark

            Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
            auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
            auto output =
                outputTensor.front()
                    .GetTensorMutableData<typename std::remove_pointer<N>::type>();
            delete[] blob;

            int embeddingSize = outputNodeDims[1] * outputNodeDims[2] * outputNodeDims[3]; // Flattened size
            result.embeddings.assign(output, output + embeddingSize); // Save embeddings

            #ifdef benchmark
            clock_t starttime_4 = clock();
            double pre_process_time = static_cast<double>(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
            double process_time = static_cast<double>(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
            double post_process_time = static_cast<double>(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
            if (cudaEnable_)
            {
                CONSOLE_BRIDGE_logInform("[SAM_encoder(CUDA)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                                         pre_process_time, process_time, post_process_time);
            }
            else
            {
                CONSOLE_BRIDGE_logInform("[SAM_encoder(CPU)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                                   pre_process_time, process_time, post_process_time);
            }
            #endif // benchmark
            break;
        }
        case SEG::SAM_SEGMENT_DECODER:
        {
            // Use embeddings from the last result
            std::vector<float> embeddings = result.embeddings;
            // Create tensor for decoder
            std::vector<int64_t> decoderInputDims = {1, 256, 64, 64}; // Adjust based on your decoder's requirements

            // Create point coordinates for testing purposes
            #ifdef ROI
            // Create a window for user interaction
            namedWindow("Select and View Result", cv::WINDOW_AUTOSIZE);

            // Let the user select the bounding box
            cv::Rect bbox = selectROI("Select and View Result", iImg, false, false);

            // Check if a valid bounding box was selected
            if (bbox.width == 0 || bbox.height == 0)
            {
              CONSOLE_BRIDGE_logError("No valid bounding box selected.");
              return "[SAM]: NO valid Box.";
            }

            std::vector<cv::Rect> boundingBoxes;
            boundingBoxes.push_back(bbox);
            #endif // ROI

            #ifdef benchmark
            clock_t starttime_2 = 0;
            clock_t starttime_3 = 0;
            #endif // benchmark

            #ifdef ROI
            for (const auto& box : boundingBoxes)
            #else
            for (const auto& box : result.boxes)
            #endif // ROI
            {
                Ort::Value decoderInputTensor = Ort::Value::CreateTensor<float>(
                    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                    embeddings.data(), // Use the embeddings from the encoder
                    embeddings.size(), // Total number of elements
                    decoderInputDims.data(), decoderInputDims.size());

                // Use center of bounding box as foreground point
                // float centerX = box.x + box.width / 2.0;
                // float centerY = box.y + box.height / 2.0;

                // Convert bounding box to points
                std::vector<float> pointCoords = {
                    static_cast<float>(box.x),
                    static_cast<float>(box.y), // Top-left
                    static_cast<float>(box.x + box.width),
                    static_cast<float>(box.y + box.height) // Bottom-right
                };

                std::vector<float> pointCoordsScaled;

                std::vector<int64_t> pointCoordsDims = {1, 2, 2}; // 2 points, each with (x, y)

                // Labels for the points
                std::vector<float> pointLabels = {2.0f, 3.0f}; // Box prompt labels
                std::vector<int64_t> pointLabelsDims = {1, 2};

                // Create dummy mask_input and has_mask_input
                std::vector<float> maskInput(256 * 256, 0.0f); // Fill with zeros
                std::vector<int64_t> maskInputDims = {1, 1, 256, 256};

                std::vector<float> hasMaskInput = {0.0f}; // No mask provided
                std::vector<int64_t> hasMaskInputDims = {1};

                utilities.ScaleBboxPoints(iImg, imgSize_, pointCoords, pointCoordsScaled);

                std::vector<Ort::Value> inputTensors = utilities.PrepareInputTensor(
                    decoderInputTensor, pointCoordsScaled, pointCoordsDims, pointLabels,
                    pointLabelsDims, maskInput, maskInputDims, hasMaskInput,
                    hasMaskInputDims);

                #ifdef benchmark
                starttime_2 = clock();
                #endif // benchmark
                auto output_tensors = session_->Run(
                    options_, inputNodeNames_.data(), inputTensors.data(),
                    inputTensors.size(), outputNodeNames_.data(), outputNodeNames_.size());

                #ifdef benchmark
                starttime_3 = clock();
                #endif // benchmark

                utilities.PostProcess(output_tensors, iImg, imgSize_, result);
            }

            // Add the result to oResult
            oResult.push_back(result);

            delete[] blob;

            #ifdef benchmark
            clock_t starttime_4 = clock();
            double pre_process_time = static_cast<double>(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
            double process_time = static_cast<double>(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
            double post_process_time = static_cast<double>(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
            if (cudaEnable_)
            {
                CONSOLE_BRIDGE_logInform("[SAM_decoder(CUDA)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                                     pre_process_time, process_time, post_process_time);
            }
            else
            {
                CONSOLE_BRIDGE_logInform("[SAM_decoder(CPU)]: %.2fms pre-process, %.2fms inference, %.2fms post-process.",
                                     pre_process_time, process_time, post_process_time);
            }
            #endif // benchmark
            break;
        }
    default:
        CONSOLE_BRIDGE_logError("[SAM]: " "Not support model type.");
  }

  return RET_OK;
}

char* SAM::WarmUpSession_(SEG::MODEL_TYPE modelType)
{
    clock_t starttime_1 = clock();
    Utils utilities;
    cv::Mat iImg = cv::Mat(cv::Size(imgSize_.at(0), imgSize_.at(1)), CV_8UC3);
    cv::Mat processedImg;
    utilities.PreProcess(iImg, imgSize_, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[iImg.total() * 3];
        utilities.BlobFromImage(processedImg, blob);
        // NCHW: H=imgSize[1], W=imgSize[0]  // FIX
        std::vector<int64_t> SAM_input_node_dims = { 1, 3, imgSize_.at(1), imgSize_.at(0) }; // FIX
        switch (modelType)
        {
            case SEG::SAM_SEGMENT_ENCODER:
            {
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                    blob, 3 * imgSize_.at(0) * imgSize_.at(1),
                    SAM_input_node_dims.data(), SAM_input_node_dims.size());
                auto output_tensors = session_->Run(options_, inputNodeNames_.data(), &input_tensor, 1,
                                                outputNodeNames_.data(), outputNodeNames_.size());
                delete[] blob;
                clock_t starttime_4 = clock();
                double post_process_time = static_cast<double>(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
                if (cudaEnable_)
                {
                    CONSOLE_BRIDGE_logInform("[SAM(CUDA)]: Cuda warm-up cost %.2f ms.", post_process_time);
                }

                break;
            }

            case SEG::SAM_SEGMENT_DECODER:
            {
                std::vector<int64_t> inputNodeDims = {1, 256, 64, 64}; // BUG: That was 236 instead of 256
                // Use embeddings from the last result
                std::vector<float> dummyEmbeddings(256 * 64 * 64, 1.0f); // Fill with zeros or any dummy values
                std::vector<int64_t> decoderInputDims = {1, 256, 64, 64}; // Adjust based on your decoder's requirements

                // Create dummy point coordinates and labels
                std::vector<cv::Rect> boundingBoxes = {
                    cv::Rect(0, 0, 100, 100), // Example bounding box with (x, y, width, height)
                    // cv::Rect(0, 0, 473, 359) // Another example bounding box
                };
                for (const auto& bbox : boundingBoxes)
                {
                    Ort::Value decoderInputTensor = Ort::Value::CreateTensor<float>(
                        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                        dummyEmbeddings.data(), // Use the embeddings from the encoder
                        dummyEmbeddings.size(), // Total number of elements
                        decoderInputDims.data(), decoderInputDims.size());
    
                    // Convert bounding box to points
                    // Use center of bounding box as foreground point
                    float centerX = bbox.x + bbox.width / 2.0;
                    float centerY = bbox.y + bbox.height / 2.0;

                    std::vector<float> pointCoords = {centerX, centerY}; // Center point (foreground)

                    std::vector<int64_t> pointCoordsDims = {1, 1, 2}; // 2 points, each with (x, y)

                    std::vector<float> pointCoordsScaled;

                    utilities.ScaleBboxPoints(iImg, imgSize_, pointCoords, pointCoordsScaled);

                    // Labels for the points
                    std::vector<float> pointLabels = {1.0f}; // All points are foreground
                    std::vector<int64_t> pointLabelsDims = {1, 1};

                    // Create dummy mask_input and has_mask_input
                    std::vector<float> maskInput(256 * 256, 0.0f); // Fill with zeros
                    std::vector<int64_t> maskInputDims = {1, 1, 256, 256};
                    std::vector<float> hasMaskInput = {0.0f}; // No mask provided
                    std::vector<int64_t> hasMaskInputDims = {1};

                    std::vector<Ort::Value> inputTensors = utilities.PrepareInputTensor(
                        decoderInputTensor, pointCoordsScaled, pointCoordsDims, pointLabels,
                        pointLabelsDims, maskInput, maskInputDims, hasMaskInput,
                        hasMaskInputDims);

                    auto output_tensors = session_->Run(
                        options_, inputNodeNames_.data(), inputTensors.data(),
                        inputTensors.size(), outputNodeNames_.data(), outputNodeNames_.size());
                }

                delete[] blob;
                clock_t starttime_4 = clock();
                double post_process_time = static_cast<double>(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
                if (cudaEnable_)
                {
                    CONSOLE_BRIDGE_logInform("[SAM(CUDA)]: Cuda warm-up cost %.2f ms.", post_process_time);
                }

                break;
            }
        }
    }

    return RET_OK;
}
