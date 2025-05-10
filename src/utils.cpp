#include "utils.h"
// Constructor
Utils::Utils(){

}

// Destructor
Utils::~Utils(){
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

    return RET_OK;
}

void Utils::ScaleBboxPoints(const cv::Mat& iImg, std::vector<int> imgSize, std::vector<float>& pointCoords, std::vector<float>& pointCoordsScaled){

    pointCoordsScaled.clear();

    // Calculate same scale as preprocessing
    float scale;
    if (iImg.cols >= iImg.rows) {
        scale = imgSize[0] / (float)iImg.cols;
        resizeScalesBbox = iImg.cols / (float)imgSize[0];
    } else {
        scale = imgSize[1] / (float)iImg.rows;
        resizeScalesBbox = iImg.rows / (float)imgSize[1];
    }

    // TOP-LEFT placement (matching PreProcess)
    for (size_t i = 0; i < pointCoords.size(); i += 2) {
        if (i + 1 < pointCoords.size()) {
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
    std::vector<int64_t> pointLabelsDims, std::vector<float>& maskInput, std::vector<int64_t> maskInputDims, std::vector<float>& hasMaskInput, std::vector<int64_t> hasMaskInputDims){

Ort::Value pointCoordsTensor = Ort::Value::CreateTensor<float>(
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
    pointCoordsScaled.data(),
    pointCoordsScaled.size(),
    pointCoordsDims.data(),
    pointCoordsDims.size()
);



Ort::Value pointLabelsTensor = Ort::Value::CreateTensor<float>(
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
    pointLabels.data(),
    pointLabels.size(),
    pointLabelsDims.data(),
    pointLabelsDims.size()
);



Ort::Value maskInputTensor = Ort::Value::CreateTensor<float>(
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
    maskInput.data(),
    maskInput.size(),
    maskInputDims.data(),
    maskInputDims.size()
);



Ort::Value hasMaskInputTensor = Ort::Value::CreateTensor<float>(
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
    hasMaskInput.data(),
    hasMaskInput.size(),
    hasMaskInputDims.data(),
    hasMaskInputDims.size()
);

// Pass all inputs to the decoder
std::vector<Ort::Value> inputTensors;
inputTensors.push_back(std::move(decoderInputTensor));
inputTensors.push_back(std::move(pointCoordsTensor));
inputTensors.push_back(std::move(pointLabelsTensor));
inputTensors.push_back(std::move(maskInputTensor));
inputTensors.push_back(std::move(hasMaskInputTensor));

return inputTensors;
}
void Utils::overlay(std::vector<Ort::Value>& output_tensors, const cv::Mat& iImg, std::vector<int> imgSize, SEG::DL_RESULT& result){
    // Process decoder output (masks)
    if (output_tensors.size() > 0)
    {
        // Get the masks from the output tensor
        auto scoresTensor = std::move(output_tensors[0]);  // IoU scores
        auto masksTensor = std::move(output_tensors[1]); // First output should be the masks PROBABLY WRONG
        auto masksInfo = masksTensor.GetTensorTypeAndShapeInfo();
        auto masksShape = masksInfo.GetShape();


        if (masksShape.size() == 4)
        {
            auto masksData = masksTensor.GetTensorMutableData<float>();
            auto scoresData = scoresTensor.GetTensorMutableData<float>();

            size_t batchSize = masksShape[0]; // Usually 1
            size_t numMasks = masksShape[1];  // Number of masks (typically 1)
            size_t height = masksShape[2];    // Height of mask
            size_t width = masksShape[3];     // Width of mask


            // Find the best mask (highest IoU score)
            float bestScore = -1;
            size_t bestMaskIndex = 0;

            for (size_t i = 0; i < numMasks; ++i)
            {

                float score = scoresData[i];

                if (score > bestScore) {
                    bestScore = score;
                    bestMaskIndex = i;
                }
            }

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

                // 1. Calculate the dimensions the image had during preprocessing
            float scale;
            int processedWidth, processedHeight;
            if (iImg.cols >= iImg.rows) {
                scale = (float)imgSize[0] / iImg.cols;
                processedWidth = imgSize[0];
                processedHeight = int(iImg.rows * scale);
            } else {
                scale = (float)imgSize[1] / iImg.rows;
                processedWidth = int(iImg.cols * scale);
                processedHeight = imgSize[1];
            }
            // 2. Resize mask to match the SAM input dimensions
            cv::Mat resizedMask;
            cv::resize(mask, resizedMask, cv::Size(256, 256));

            // 3. Extract the portion that corresponds to the actual image (no padding)
            int cropWidth = std::min(256, int(256 * processedWidth / (float)imgSize[0]));
            int cropHeight = std::min(256, int(256 * processedHeight / (float)imgSize[1]));
            cv::Mat croppedMask = resizedMask(cv::Rect(0, 0, cropWidth, cropHeight));

            // 4. Resize directly to original image dimensions in one step
            cv::Mat finalMask;
            cv::resize(croppedMask, finalMask, cv::Size(iImg.cols, iImg.rows));


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


            // Visualize the mask on the input image
            cv::Mat colorMask = cv::Mat::zeros(iImg.size(), CV_8UC3);
            colorMask.setTo(cv::Scalar(255, 0, 0), finalMask); // Red color for mask

            // Blend the original image with the colored mask
            cv::addWeighted(iImg, 1, colorMask, 0.9, 0.6, iImg);

            // Save or display the result
            cv::imwrite("segmentation_result_" + std::to_string(bestMaskIndex) + ".jpg", iImg);
            cv::imwrite("mask_" + std::to_string(bestMaskIndex) + ".jpg", finalMask);
            }else
            {
                std::cerr << "[SAM]: Unexpected mask tensor shape." << std::endl;
            }
        }else
            {
                std::cerr << "[SAM]: No masks found in the output tensor." << std::endl;
            }
    }