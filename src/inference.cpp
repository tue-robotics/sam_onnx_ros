#include "inference.h"
#include "utils.h"
#include <regex>
#include <typeinfo>

#define benchmark
// #define min(a,b)            (((a) < (b)) ? (a) : (b))

SAM::SAM() {

}


SAM::~SAM() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


const char* SAM::CreateSession(DL_INIT_PARAM& iParams) {
    const char* Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        Ret = "[SAM]:Your model path is error.Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try
    {
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        cudaEnable = iParams.cudaEnable;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Sam");
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

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        inputNodeNames.clear();
        outputNodeNames.clear();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };

        //std::vector<long int> input_shape;
        //std::vector<long int> output_shape;
        //size_t input_tensor_size = 0;
        //size_t output_tensor_size = 0;
        //Get input and output tensor size

        //auto input_tensor_size = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount();
        //auto output_tensor_size = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount();
        auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        auto output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        auto output_type = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
        std::cout << "output_type: " << output_type << std::endl;
        //std::cout << "input_tensor_size: " << input_tensor_size << std::endl;
        //std::cout << "output_tensor_size: " << output_tensor_size << std::endl;
        std::cout << "input_shape: ";
        for (auto i : input_shape)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        std::cout << "output_shape: ";
        for (auto i : output_shape)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        WarmUpSession(modelType);
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[SAM]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[SAM]:Create session failed.";
    }

}

const char* SAM::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult, MODEL_TYPE modelType) {
    #ifdef benchmark
        clock_t starttime_1 = clock();
    #endif // benchmark
        Utils utilities;
        const char* Ret = RET_OK;
        cv::Mat processedImg;
        utilities.PreProcess(iImg, imgSize, processedImg);
        if (modelType < 4)
        {
            float* blob = new float[processedImg.total() * 3];
            utilities.BlobFromImage(processedImg, blob);
            std::vector<int64_t> inputNodeDims;
            if (modelType == SAM_SEGMENT_ENCODER)
            {
                inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
            }
            else if (modelType == SAM_SEGMENT_DECODER)
            {
                // For SAM decoder model, the input size is different
                // Assuming the input size is 236x64x64 for the decoder
                // You can adjust this based on your actual model requirements
                // For example, if the input size is 1x3x236x64, you can set it as follows:
                // inputNodeDims = { 1, 3, 236, 64 };
                // But here we are using 1x236x64x64 as per your original code

                inputNodeDims = { 1, 256, 64, 64 };
            }
            TensorProcess(starttime_1, iImg, blob, inputNodeDims, modelType, oResult, utilities);
        }
        else
        {
    #ifdef USE_CUDA
            half* blob = new half[processedImg.total() * 3];
            BlobFromImage(processedImg, blob);
            std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
            TensorProcess(starttime_1, iImg, blob, inputNodeDims, modelType, oResult, utilities);
    #endif
        }

        return Ret;
    }

    template<typename N>
    char* SAM::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        MODEL_TYPE modelType, std::vector<DL_RESULT>& oResult, Utils& utilities) {

        switch (modelType)
        {
        case SAM_SEGMENT_ENCODER:
        // case OTHER_SAM_MODEL:
        {

            Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
                inputNodeDims.data(), inputNodeDims.size());
        #ifdef benchmark
            clock_t starttime_2 = clock();
        #endif // benchmark
            auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
                outputNodeNames.size());
        #ifdef benchmark
            clock_t starttime_3 = clock();
        #endif // benchmark

            Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
            auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
            auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
            //std::vector<int64_t> outputNodeDims = outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
            delete[] blob;

            DL_RESULT result;
            int embeddingSize = outputNodeDims[1] * outputNodeDims[2] * outputNodeDims[3]; // Flattened size
            result.embeddings.assign(output, output + embeddingSize); // Save embeddings
            oResult.push_back(result);


    #ifdef benchmark
            clock_t starttime_4 = clock();
            double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
            double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
            double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
            if (cudaEnable)
            {
                std::cout << "[SAM(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
            }
            else
            {
                std::cout << "[SAM(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
            }
    #endif // benchmark

            break;
        }
        case SAM_SEGMENT_DECODER:
        //case <OTHER MODEL>:
        {
            // Use embeddings from the last result
            std::vector<float> embeddings = oResult.back().embeddings;
            // Create tensor for decoder
            std::vector<int64_t> decoderInputDims = { 1, 256, 64, 64 }; // Adjust based on your decoder's requirements
            Ort::Value decoderInputTensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                embeddings.data(), // Use the embeddings from the encoder
                embeddings.size(), // Total number of elements
                decoderInputDims.data(),
                decoderInputDims.size()
            );

            /////////////////// DEBUG /////////////////////

            std::cout << "Decoder Input Tensor Shape: ";
            for (auto dim : decoderInputTensor.GetTensorTypeAndShapeInfo().GetShape()) {
                std::cout << dim << " " << std::endl;
            }

            if (oResult.empty()) {
                std::cerr << "[SAM]: No embeddings available from the encoder." << std::endl;
                return "[SAM]: Decoder failed due to missing embeddings.";
            }
            std::cout << "Embeddings size: " << embeddings.size() << std::endl;
            std::cout << "Creating decoderInputTensor with dimensions: ";
            for (auto dim : decoderInputDims) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "Input Node Names:" << std::endl;
            for (const auto& name : inputNodeNames) {
                std::cout << name << std::endl;
            }
            if (embeddings.size() != 256 * 64 * 64) {
                std::cerr << "[SAM]: Embeddings size mismatch. Expected 256*64*64, got " << embeddings.size() << std::endl;
                return "[SAM]: Decoder failed due to invalid embeddings.";
            }
            if (!decoderInputTensor.IsTensor()) {
                std::cerr << "[SAM]: Failed to create decoderInputTensor." << std::endl;
                return "[SAM]: Tensor creation failed.";
            }

            std::cout << "First 10 values of embeddings: ";
            for (size_t i = 0; i < 10; ++i) {
                std::cout << embeddings[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "Embeddings data pointer: " << static_cast<const void*>(embeddings.data()) << std::endl;

            if (embeddings.size() != 256 * 64 * 64) {
                std::cerr << "[SAM]: Embeddings size mismatch. Expected 256*64*64, got " << embeddings.size() << std::endl;
                return "[SAM]: Decoder failed due to invalid embeddings.";
            }

            std::cout << "Input Node Names and Corresponding Tensors:" << std::endl;
            for (size_t i = 0; i < inputNodeNames.size(); ++i) {
                std::cout << inputNodeNames[i] << " -> Tensor " << i << std::endl;
            }

            // Debug: Print the first few values of decoderInputTensor
            float* tensorData = decoderInputTensor.GetTensorMutableData<float>();
            std::cout << "First 10 values of decoderInputTensor: ";
            for (size_t i = 0; i < 10; ++i) {
                std::cout << tensorData[i] << " ";
            }
            std::cout << std::endl;

            // Debug: Print the data pointer of decoderInputTensor
            std::cout << "decoderInputTensor data pointer: " << static_cast<const void*>(tensorData) << std::endl;

            /////////////////// DEBUG STOP /////////////////////

            // Create  point coordinates and labels


            // Create a window for user interaction
            namedWindow("Select and View Result", cv::WINDOW_AUTOSIZE);

            // Let the user select the bounding box
            cv::Rect bbox = selectROI("Select and View Result", iImg, false, false);

            // Check if a valid bounding box was selected
            if (bbox.width == 0 || bbox.height == 0)
            {
                std::cerr << "No valid bounding box selected." << std::endl;
                return "[SAM]: NO valid Box.";
            }

            // cv::Rect bbox1(100, 30, 280, 320);
            //cv::Rect bbox1(138, 29, 170, 301);
            std::vector<cv::Rect> boundingBoxes;
            boundingBoxes.push_back(bbox);

            for (const auto &bbox : boundingBoxes)
            {
                // Use center of bounding box as foreground point
                float centerX = bbox.x + bbox.width/2;
                float centerY = bbox.y + bbox.height/2;

                std::vector<float> pointCoords = {
                    centerX, centerY  // Center point (foreground)
                };


                std::vector<float> pointCoordsScaled;

                std::vector<int64_t> pointCoordsDims = {1, 1, 2}; // 2 points, each with (x, y)

                // Labels for the points
                std::vector<float> pointLabels = {1.0f}; // All points are foreground
                std::vector<int64_t> pointLabelsDims = {1, 1};

                // Create dummy mask_input and has_mask_input
                std::vector<float> maskInput(256 * 256, 0.0f); // Fill with zeros
                std::vector<int64_t> maskInputDims = {1, 1, 256, 256};


                std::vector<float> hasMaskInput = {0.0f}; // No mask provided
                std::vector<int64_t> hasMaskInputDims = {1};

                utilities.ScaleBboxPoints(iImg, imgSize, pointCoords, pointCoordsScaled);




                std::vector<Ort::Value> inputTensors  = utilities.PrepareInputTensor(
                    decoderInputTensor,
                    pointCoordsScaled,
                    pointCoordsDims,
                    pointLabels,
                    pointLabelsDims,
                    maskInput,
                    maskInputDims,
                    hasMaskInput,
                    hasMaskInputDims
                );

            #ifdef benchmark
                starttime_2 = clock();
            #endif // benchmark
                auto output_tensors = session->Run(
                    options,
                    inputNodeNames.data(),
                    inputTensors.data(),
                    inputTensors.size(),
                    outputNodeNames.data(),
                    outputNodeNames.size());

                // Process decoder output (masks)
                if (output_tensors.size() > 0)
                {
                    // Get the masks from the output tensor
                    auto scoresTensor = std::move(output_tensors[0]);  // IoU scores
                    auto masksTensor = std::move(output_tensors[1]); // First output should be the masks PROBABLY WRONG
                    auto masksInfo = masksTensor.GetTensorTypeAndShapeInfo();
                    auto masksShape = masksInfo.GetShape();

                    // Debug print mask shape
                    std::cout << "Masks Tensor Shape: ";
                    for (auto dim : masksShape)
                    {
                        std::cout << dim << " ";
                    }
                    std::cout << std::endl;


                    if (masksShape.size() == 4)
                    {
                        auto masksData = masksTensor.GetTensorMutableData<float>();
                        auto scoresData = scoresTensor.GetTensorMutableData<float>();

                        size_t batchSize = masksShape[0]; // Usually 1
                        size_t numMasks = masksShape[1];  // Number of masks (typically 1)
                        size_t height = masksShape[2];    // Height of mask
                        size_t width = masksShape[3];     // Width of mask

                        std::cout << "Processing " << numMasks << " masks..." << std::endl;

                        // Find the best mask (highest IoU score)
                        float bestScore = -1;
                        size_t bestMaskIndex = 0;

                        for (size_t i = 0; i < numMasks; ++i)
                        {

                            float score = scoresData[i];
                            std::cout << "Mask " << i << " score: " << score << std::endl;

                            if (score > bestScore) {
                                bestScore = score;
                                bestMaskIndex = i;
                            }
                        }
                            std::cout << "Selected best mask: " << bestMaskIndex << " with score: " << bestScore << std::endl;

                            // Create OpenCV Mat for the mask
                            cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);

                            // Convert float mask to binary mask
                            for (size_t h = 0; h < height; ++h)
                            {
                                for (size_t w = 0; w < width; ++w)
                                {
                                    size_t idx = (bestMaskIndex * height * width) + (h * width) + w;
                                    float value = masksData[idx];
                                    mask.at<uchar>(h, w) = (value > 0.5f) ? 255 : 0; // Threshold at 0.5
                                }
                            }

                            // Resize mask to original image size (accounting for any scaling)
                            int limX, limY;
                            int size = std::max(iImg.rows, iImg.cols);

                            // Calculate limits for upscaling based on target dimensions
                            if (iImg.rows > iImg.cols)
                            {
                                limX = size;
                                limY = size * iImg.cols / iImg.rows;
                            }
                            else
                            {
                                limX = size * iImg.rows / iImg.cols;
                                limY = size;
                            }

                            // Ensure limX and limY do not exceed the dimensions of the mask
                            limX = std::min(limX, mask.cols);
                            limY = std::min(limY, mask.rows);

                            // First, resize the mask to model input dimensions (1024x1024)
                            cv::Mat preprocessedSizeMask;
                            cv::resize(mask, preprocessedSizeMask, cv::Size(imgSize.at(0), imgSize.at(1)));
                            // Calculate the dimensions of the actual image in the preprocessed space
                            int effectiveWidth, effectiveHeight;
                            if (iImg.cols >= iImg.rows) {
                                effectiveWidth = imgSize.at(0);  // Full width (1024)
                                effectiveHeight = int(iImg.rows / resizeScales);  // Scaled height
                            } else {
                                effectiveWidth = int(iImg.cols / resizeScales);  // Scaled width
                                effectiveHeight = imgSize.at(1);  // Full height (1024)
                            }

                            // Create mask for original image
                            cv::Mat finalMask = cv::Mat::zeros(iImg.rows, iImg.cols, CV_8UC1);

                            // Extract active area (no padding) and resize to original dimensions
                            cv::Mat activeAreaMask = preprocessedSizeMask(cv::Rect(0, 0, effectiveWidth, effectiveHeight));
                            cv::resize(activeAreaMask, finalMask, cv::Size(iImg.cols, iImg.rows));

                            // Resize the mask to the target dimensions
                            cv::resize(mask(cv::Rect(0, 0, limX, limY)), mask, cv::Size(iImg.cols, iImg.rows));
                            // cv::resize(mask, mask, cv::Size(iImg.cols, iImg.rows));

                            // Create or update a result
                            DL_RESULT result;

                            // If we want to preserve the embeddings from the encoder
                            if (!oResult.empty())
                            {
                                result.embeddings = oResult.back().embeddings;
                            }

                            // Add the mask to the result
                            result.masks.push_back(finalMask);

                            /*// Add IoU scores if available (typically second tensor)
                            if (output_tensors.size() > 1) {
                                auto scoresTensor = std::move(output_tensors[1]);
                                auto scoresData = scoresTensor.GetTensorMutableData<float>();
                                if (i < scoresTensor.GetTensorTypeAndShapeInfo().GetShape()[1]) {
                                    result.confidence = scoresData[i];
                                    std::cout << "Mask confidence: " << result.confidence << std::endl;
                                }
                            }*/

                            // Add the result to oResult
                            oResult.push_back(result);

                            // Visualize the mask on the input image
                            cv::Mat colorMask = cv::Mat::zeros(iImg.size(), CV_8UC3);
                            colorMask.setTo(cv::Scalar(0, 0, 255), finalMask); // Red color for mask

                            // Blend the original image with the colored mask
                            cv::addWeighted(iImg, 1, colorMask, 0.3, 0.0, iImg);

                            // Save or display the result
                            cv::imwrite("segmentation_result_" + std::to_string(bestMaskIndex) + ".jpg", iImg);
                            cv::imwrite("mask_" + std::to_string(bestMaskIndex) + ".jpg", finalMask);

                    }
                    else
                    {
                        std::cerr << "[SAM]: Unexpected mask tensor shape." << std::endl;
                    }
                }
            }
            // Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
            // Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            // blob, 3 * imgSize.at(0) * imgSize.at(1),
            // inputNodeDims.data(),
            // inputNodeDims.size());
            // Run the decoder session
            // auto decoderOutputTensor = decoderSession->Run(
            // Ort::RunOptions{ nullptr },
            // decoderInputNodeNames.data(),
            //&decoderInputTensor,
            // 1,
            // decoderOutputNodeNames.data(),
            // decoderOutputNodeNames.size()
            //);

            // Process decoder output (if needed)
            break;
        }
        default:
            std::cout << "[SAM]: " << "Not support model type." << std::endl;
        }
        return RET_OK;

    }


char* SAM::WarmUpSession(MODEL_TYPE modelType) {
    clock_t starttime_1 = clock();
    Utils utilities;
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    utilities.PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[iImg.total() * 3];
        utilities.BlobFromImage(processedImg, blob);
        std::vector<int64_t> SAM_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        switch (modelType)
        {
        case SAM_SEGMENT_ENCODER: {
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
                SAM_input_node_dims.data(), SAM_input_node_dims.size());
            auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
                outputNodeNames.size());
            delete[] blob;
            clock_t starttime_4 = clock();
            double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
            if (cudaEnable)
            {
                std::cout << "[SAM(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
            }
            break;
        }

        case SAM_SEGMENT_DECODER: {
            std::vector<int64_t> inputNodeDims = { 1, 256, 64, 64 }; // BUG: That was 236 instead of 256
            // Use embeddings from the last result
            std::vector<float> dummyEmbeddings(256 * 64 * 64, 1.0f); // Fill with zeros or any dummy values
            std::vector<int64_t> decoderInputDims = { 1, 256, 64, 64 }; // Adjust based on your decoder's requirements
            Ort::Value decoderInputTensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                dummyEmbeddings.data(), // Use the embeddings from the encoder
                dummyEmbeddings.size(), // Total number of elements
                decoderInputDims.data(),
                decoderInputDims.size()
            );

            // Create dummy point coordinates and labels
            std::vector<cv::Rect> boundingBoxes = {
                // cv::Rect(0, 0, 100, 100), // Example bounding box with (x, y, width, height)
                cv::Rect(0, 0, 473, 359) // Another example bounding box
            };
            for (const auto& bbox : boundingBoxes) {
                // Convert bounding box to points
                // Use center of bounding box as foreground point
                float centerX = bbox.x + bbox.width/2;
                float centerY = bbox.y + bbox.height/2;

                std::vector<float> pointCoords = {
                    centerX, centerY  // Center point (foreground)
                };

                std::vector<int64_t> pointCoordsDims = { 1, 1, 2 }; // 2 points, each with (x, y)

                std::vector<float> pointCoordsScaled;

                utilities.ScaleBboxPoints(iImg, imgSize, pointCoords, pointCoordsScaled);

                // Labels for the points
                std::vector<float> pointLabels = {1.0f}; // All points are foreground
                std::vector<int64_t> pointLabelsDims = { 1, 1};
                // Create dummy mask_input and has_mask_input
                std::vector<float> maskInput(256 * 256, 0.0f); // Fill with zeros
                std::vector<int64_t> maskInputDims = { 1, 1, 256, 256 };
                std::vector<float> hasMaskInput = { 0.0f }; // No mask provided
                std::vector<int64_t> hasMaskInputDims = { 1 };

                std::vector<Ort::Value> inputTensors  = utilities.PrepareInputTensor(
                    decoderInputTensor,
                    pointCoordsScaled,
                    pointCoordsDims,
                    pointLabels,
                    pointLabelsDims,
                    maskInput,
                    maskInputDims,
                    hasMaskInput,
                    hasMaskInputDims
                );

                auto output_tensors = session->Run(
                    options,
                    inputNodeNames.data(),
                    inputTensors.data(),
                    inputTensors.size(),
                    outputNodeNames.data(),
                    outputNodeNames.size()
                ); }

            outputNodeNames.size();
            delete[] blob;
            clock_t starttime_4 = clock();
            double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
            if (cudaEnable)
            {
                std::cout << "[SAM(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
            }

            break;
        }
    }

    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> SAM_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), SAM_input_node_dims.data(), SAM_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[SAM(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    return RET_OK;
}
