#include "inference.h"
#include <regex>

#define benchmark
#define min(a,b)            (((a) < (b)) ? (a) : (b))

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


// Definition: Flattened image to blob (and normalizaed) for deep learning inference. Also reorganize from HWC to CHW.
// Note: Not in the header file since it is not used outside of this file.
template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}

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
        WarmUpSession();
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

const char* SAM::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult, std::string& modelPath) {
    #ifdef benchmark
        clock_t starttime_1 = clock();
    #endif // benchmark

        const char* Ret = RET_OK;
        cv::Mat processedImg;
        PreProcess(iImg, imgSize, processedImg);
        if (modelType < 4)
        {
            float* blob = new float[processedImg.total() * 3];
            BlobFromImage(processedImg, blob);
            std::vector<int64_t> inputNodeDims;
            if (modelPath == "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_encoder.onnx")
            {
                inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
            }
            else if (modelPath == "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_mask_decoder.onnx")
            {
                inputNodeDims = { 1, 236, 64, 64 };
            }
            TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
        }
        else
        {
    #ifdef USE_CUDA
            half* blob = new half[processedImg.total() * 3];
            BlobFromImage(processedImg, blob);
            std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
            TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    #endif
        }

        return Ret;
    }

    template<typename N>
    char* SAM::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult) {
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
        switch (modelType)
        {
        case SAM_SEGMENT_ENCODER:
        // case OTHER_SAM_MODEL:
        {

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
                std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
            }
            else
            {
                std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
            }
    #endif // benchmark

            break;
        }
        case YOLO_CLS:
        case YOLO_CLS_HALF:
        {
            break;
        }
        default:
            std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
        }
        return RET_OK;

    }


char* SAM::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    switch (modelType)
    {
    case SAM_SEGMENT_ENCODER:
    {
        if (iImg.cols >= iImg.rows)
        {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else
        {
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
        break;
    }
    // you can add more preprocess methods here for different SAM models
    }
    return RET_OK;
}


char* SAM::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> SAM_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
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
