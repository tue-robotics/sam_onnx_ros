#include "utils.h"
// Constructor
Utils::Utils(){

}

// Destructor
Utils::~Utils(){
}

char* Utils::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
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

void Utils::ScaleBboxPoints(cv::Mat& iImg, std::vector<int> imgSize, std::vector<float>& pointCoords, std::vector<float>& pointCoordsScaled){

    // For landscape images (width >= height)
    if (iImg.cols >= iImg.rows) {
        resizeScalesBbox = iImg.cols / (float)imgSize.at(0);
    }
    // For portrait images (height > width)
    else {
        resizeScalesBbox = iImg.rows / (float)imgSize.at(0);
    }

    for (auto i : pointCoords)
    {
        pointCoordsScaled.push_back(i / resizeScalesBbox);
    };
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
void Utils::overlay(std::vector<Ort::Value>& output_tensors, cv::Mat& iImg, std::vector<int> imgSize, std::vector<DL_RESULT>& oResult){
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
                    effectiveHeight = int(iImg.rows / resizeScalesBbox);  // Scaled height
                } else {
                    effectiveWidth = int(iImg.cols / resizeScalesBbox);  // Scaled width
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
            }else
            {
                std::cerr << "[SAM]: Unexpected mask tensor shape." << std::endl;
            }
        }else
            {
                std::cerr << "[SAM]: No masks found in the output tensor." << std::endl;
            }
    }