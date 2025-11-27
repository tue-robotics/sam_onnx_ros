#include "sam_onnx_ros/utils.hpp"

// #define LOGGING

// Constructor
Utils::Utils()
{
}

// Destructor
Utils::~Utils()
{
}

char* Utils::PreProcess(const cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
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

    if (iImg.cols >= iImg.rows)
    {
        // Width-dominant: scale by target width (iImgSize[0])
        resizeScales_ = iImg.cols / static_cast<float>(iImgSize.at(0));
        // Resize to target width, scaling height to maintain aspect ratio
        cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), static_cast<int>(iImg.rows / resizeScales_)));
    }
    else
    {
        // Height-dominant: scale by target height (iImgSize[1])
        resizeScales_ = iImg.rows / static_cast<float>(iImgSize.at(1));
        // Resize width proportionally to target height to maintain aspect ratio (height-dominant case)
        cv::resize(oImg, oImg, cv::Size(static_cast<int>(iImg.cols / resizeScales_), iImgSize.at(1)));
    }

    // Letterbox top-left into a canvas of size (H=iImgSize[1], W=iImgSize[0])
    cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(1), iImgSize.at(0), CV_8UC3);
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    oImg = tempImg;

    return RET_OK;
}

void Utils::ScaleBboxPoints(const cv::Mat& iImg, std::vector<int> imgSize, std::vector<float>& pointCoords, std::vector<float>& pointCoordsScaled)
{

    pointCoordsScaled.clear();

    // Calculate same scale as preprocessing
    float scale;
    if (iImg.cols >= iImg.rows)
    {
        scale = imgSize[0] / (float)iImg.cols;
        resizeScalesBbox_ = iImg.cols / (float)imgSize[0];
    }
    else
    {
        scale = imgSize[1] / (float)iImg.rows;
        resizeScalesBbox_ = iImg.rows / (float)imgSize[1];
    }

    // Top-Left placement (matching PreProcess)
    for (size_t i = 0; i < pointCoords.size(); i += 2)
    {
        if (i + 1 < pointCoords.size())
        {
            float x = pointCoords[i];
            float y = pointCoords[i + 1];

            // Simply scale coordinates - NO padding addition
            float scaledX = x * scale;
            float scaledY = y * scale;

            pointCoordsScaled.push_back(scaledX);
            pointCoordsScaled.push_back(scaledY);
        }
    }
}

std::vector<Ort::Value> Utils::PrepareInputTensor(Ort::Value& decoderInputTensor, std::vector<float>& pointCoordsScaled, std::vector<int64_t> pointCoordsDims, std::vector<float>& pointLabels,
                                                  std::vector<int64_t> pointLabelsDims, std::vector<float>& maskInput, std::vector<int64_t> maskInputDims, std::vector<float>& hasMaskInput, std::vector<int64_t> hasMaskInputDims)
{

    Ort::Value pointCoordsTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        pointCoordsScaled.data(),
        pointCoordsScaled.size(),
        pointCoordsDims.data(),
        pointCoordsDims.size());

    Ort::Value pointLabelsTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        pointLabels.data(),
        pointLabels.size(),
        pointLabelsDims.data(),
        pointLabelsDims.size());

    Ort::Value maskInputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        maskInput.data(),
        maskInput.size(),
        maskInputDims.data(),
        maskInputDims.size());

    Ort::Value hasMaskInputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        hasMaskInput.data(),
        hasMaskInput.size(),
        hasMaskInputDims.data(),
        hasMaskInputDims.size());

    // Pass all inputs to the decoder
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(decoderInputTensor));
    inputTensors.push_back(std::move(pointCoordsTensor));
    inputTensors.push_back(std::move(pointLabelsTensor));
    inputTensors.push_back(std::move(maskInputTensor));
    inputTensors.push_back(std::move(hasMaskInputTensor));

    return inputTensors;
}
void Utils::PostProcess(std::vector<Ort::Value>& output_tensors, const cv::Mat& iImg, std::vector<int> imgSize, SEG::DL_RESULT& result)
{
    if (output_tensors.empty())
    {
        std::cerr << "[SAM]: Decoder returned no outputs." << std::endl;
        return;
    }

    // Detect masks (4D) and scores (1D/2D) by shape
    int masksIdx = -1, scoresIdx = -1;
    for (int i = 0; i < static_cast<int>(output_tensors.size()); ++i)
    {
        const auto& val = output_tensors[i];
        auto shape = val.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4) masksIdx = i;
        else if (shape.size() <= 2) scoresIdx = i;
    }

    if (masksIdx < 0)
    {
        std::cerr << "[SAM]: No 4D mask tensor found in decoder outputs." << std::endl;
        return;
    }

    auto masksTensor = std::move(output_tensors[masksIdx]);
    const float* scoresData = nullptr;
    if (scoresIdx >= 0)
    {
        scoresData = output_tensors[scoresIdx].GetTensorMutableData<float>();
    }

    auto masksInfo  = masksTensor.GetTensorTypeAndShapeInfo();
    auto masksShape = masksInfo.GetShape();

    if (masksShape.size() == 4)
    {
        auto masksData  = masksTensor.GetTensorMutableData<float>();

        const size_t numMasks = static_cast<size_t>(masksShape[1]);
        const size_t height   = static_cast<size_t>(masksShape[2]);
        const size_t width    = static_cast<size_t>(masksShape[3]);

        // Pick best mask by score if available
        float bestScore = -1.0f;
        size_t bestMaskIndex = 0;
        if (scoresData)
        {
            for (size_t i = 0; i < numMasks; ++i)
            {
                const float s = scoresData[i];
                if (s > bestScore) { bestScore = s; bestMaskIndex = i; }
            }
        }

        // Compute preprocessed region (top-left anchored) to undo letterbox
        float scale;
        int processedWidth, processedHeight;
        if (iImg.cols >= iImg.rows)
        {
            scale = static_cast<float>(imgSize[0]) / static_cast<float>(iImg.cols);
            processedWidth  = imgSize[0];
            processedHeight = static_cast<int>(iImg.rows * scale);
        }
        else
        {
            scale = static_cast<float>(imgSize[1]) / static_cast<float>(iImg.rows);
            processedWidth  = static_cast<int>(iImg.cols * scale);
            processedHeight = imgSize[1];
        }

        auto clampDim = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };

        // Wrap selected mask plane as float prob map
        const size_t planeOffset = bestMaskIndex * height * width;
        cv::Mat prob32f(static_cast<int>(height), static_cast<int>(width), CV_32F, const_cast<float*>(masksData + planeOffset));

        // Crop padding region in mask space
        const int cropW = clampDim(static_cast<int>(std::round(static_cast<float>(width)  * processedWidth  / static_cast<float>(imgSize[0]))), 1, static_cast<int>(width));
        const int cropH = clampDim(static_cast<int>(std::round(static_cast<float>(height) * processedHeight / static_cast<float>(imgSize[1]))), 1, static_cast<int>(height));
        cv::Mat probCropped = prob32f(cv::Rect(0, 0, cropW, cropH));

        // Resize to original image size and threshold
        cv::Mat probResized;
        cv::resize(probCropped, probResized, cv::Size(iImg.cols, iImg.rows), 0, 0, cv::INTER_LINEAR);

        cv::Mat finalMask;
        cv::compare(probResized, 0.5f, finalMask, cv::CMP_GT); // CV_8U 0/255

        // Optional cleanup
        int kernelSize = std::max(5, std::min(iImg.cols, iImg.rows) / 100);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
        cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);
        cv::threshold(finalMask, finalMask, 127, 255, cv::THRESH_BINARY);

        // Save mask
        result.masks.push_back(finalMask);

        // Overlay for display on a copy (iImg is const)
        #ifdef LOGGING
        cv::Mat overlay = iImg.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(finalMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat colorMask = cv::Mat::zeros(overlay.size(), CV_8UC3);
        colorMask.setTo(cv::Scalar(0, 200, 0), finalMask);
        cv::addWeighted(overlay, 0.7, colorMask, 0.3, 0, overlay);
        cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 255), 2);

        cv::imshow("SAM Segmentation", overlay);
        cv::waitKey(0);
        cv::destroyAllWindows();
        #endif // LOGGING
    }
    else
    {
        std::cerr << "[SAM]: Unexpected mask tensor shape." << std::endl;
    }
}
