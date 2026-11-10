#include "sam_onnx_ros/utils.hpp"

#include <console_bridge/console.h>

// Constructor
Utils::Utils() {}

// Destructor
Utils::~Utils() {}

char* Utils::preProcess(const cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
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
    cv::Mat temp_img = cv::Mat::zeros(iImgSize.at(1), iImgSize.at(0), CV_8UC3);
    oImg.copyTo(temp_img(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    oImg = temp_img;

    return RET_OK;
}

void Utils::scaleBboxPoints(const cv::Mat& iImg,
                            std::vector<int> imgSize,
                            std::vector<float>& pointCoords,
                            std::vector<float>& pointCoordsScaled)
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

    // Top-Left placement (matching preProcess)
    for (size_t i = 0; i < pointCoords.size(); i += 2)
    {
        if (i + 1 < pointCoords.size())
        {
            float x = pointCoords[i];
            float y = pointCoords[i + 1];

            // Simply scale coordinates - NO padding addition
            float scaled_x = x * scale;
            float scaled_y = y * scale;

            pointCoordsScaled.push_back(scaled_x);
            pointCoordsScaled.push_back(scaled_y);
        }
    }
}

std::vector<Ort::Value> Utils::prepareInputTensor(Ort::Value& decoderInputTensor,
                                                  std::vector<float>& pointCoordsScaled,
                                                  std::vector<int64_t> pointCoordsDims,
                                                  std::vector<float>& pointLabels,
                                                  std::vector<int64_t> pointLabelsDims,
                                                  std::vector<float>& maskInput,
                                                  std::vector<int64_t> maskInputDims,
                                                  std::vector<float>& hasMaskInput,
                                                  std::vector<int64_t> hasMaskInputDims)
{

    Ort::Value point_coords_tensor =
        Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                        pointCoordsScaled.data(),
                                        pointCoordsScaled.size(),
                                        pointCoordsDims.data(),
                                        pointCoordsDims.size());

    Ort::Value point_labels_tensor =
        Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                        pointLabels.data(),
                                        pointLabels.size(),
                                        pointLabelsDims.data(),
                                        pointLabelsDims.size());

    Ort::Value mask_input_tensor =
        Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                        maskInput.data(),
                                        maskInput.size(),
                                        maskInputDims.data(),
                                        maskInputDims.size());

    Ort::Value has_mask_input_tensor =
        Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                        hasMaskInput.data(),
                                        hasMaskInput.size(),
                                        hasMaskInputDims.data(),
                                        hasMaskInputDims.size());

    // Pass all inputs to the decoder
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(decoderInputTensor));
    input_tensors.push_back(std::move(point_coords_tensor));
    input_tensors.push_back(std::move(point_labels_tensor));
    input_tensors.push_back(std::move(mask_input_tensor));
    input_tensors.push_back(std::move(has_mask_input_tensor));

    return input_tensors;
}
void Utils::postProcess(std::vector<Ort::Value>& output_tensors,
                        const cv::Mat& iImg,
                        std::vector<int> imgSize,
                        SEG::DlResult& result)
{
    if (output_tensors.empty())
    {
        CONSOLE_BRIDGE_logError("[SAM]: Decoder returned no outputs.");
        // Push empty mask to preserve index alignment with boxes (one mask slot per box).
        result.masks.push_back(cv::Mat());
        return;
    }

    // Detect masks (4D) and scores (1D/2D) by shape
    int masks_idx = -1, scores_idx = -1;
    for (int i = 0; i < static_cast<int>(output_tensors.size()); ++i)
    {
        const auto& val = output_tensors[i];
        auto shape = val.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4)
            masks_idx = i;
        else if (shape.size() <= 2)
            scores_idx = i;
    }

    if (masks_idx < 0)
    {
        CONSOLE_BRIDGE_logError("[SAM]: No 4D mask tensor found in decoder outputs.");
        // Push empty mask to preserve index alignment with boxes (one mask slot per box).
        result.masks.push_back(cv::Mat());
        return;
    }

    auto masks_tensor = std::move(output_tensors[masks_idx]);
    const float* scores_data = nullptr;
    if (scores_idx >= 0)
    {
        scores_data = output_tensors[scores_idx].GetTensorMutableData<float>();
    }

    auto masks_info = masks_tensor.GetTensorTypeAndShapeInfo();
    auto masks_shape = masks_info.GetShape();

    if (masks_shape.size() == 4)
    {
        auto masks_data = masks_tensor.GetTensorMutableData<float>();

        const size_t num_masks = static_cast<size_t>(masks_shape[1]);
        const size_t height = static_cast<size_t>(masks_shape[2]);
        const size_t width = static_cast<size_t>(masks_shape[3]);

        // Pick best mask by score if available
        float best_score = -1.0f;
        size_t best_mask_index = 0;
        if (scores_data)
        {
            for (size_t i = 0; i < num_masks; ++i)
            {
                const float s = scores_data[i];
                if (s > best_score)
                {
                    best_score = s;
                    best_mask_index = i;
                }
            }
        }

        // Compute preprocessed region (top-left anchored) to undo letterbox
        float scale;
        int processed_width, processed_height;
        if (iImg.cols >= iImg.rows)
        {
            scale = static_cast<float>(imgSize[0]) / static_cast<float>(iImg.cols);
            processed_width = imgSize[0];
            processed_height = static_cast<int>(iImg.rows * scale);
        }
        else
        {
            scale = static_cast<float>(imgSize[1]) / static_cast<float>(iImg.rows);
            processed_width = static_cast<int>(iImg.cols * scale);
            processed_height = imgSize[1];
        }

        auto clamp_dim = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };

        // Wrap selected mask plane as float prob map
        const size_t plane_offset = best_mask_index * height * width;
        cv::Mat prob32f(
            static_cast<int>(height), static_cast<int>(width), CV_32F, const_cast<float*>(masks_data + plane_offset));

        // Crop padding region in mask space
        const int crop_w = clamp_dim(
            static_cast<int>(std::round(static_cast<float>(width) * processed_width / static_cast<float>(imgSize[0]))),
            1,
            static_cast<int>(width));
        const int crop_h = clamp_dim(
            static_cast<int>(std::round(static_cast<float>(height) * processed_height / static_cast<float>(imgSize[1]))),
            1,
            static_cast<int>(height));
        cv::Mat prob_cropped = prob32f(cv::Rect(0, 0, crop_w, crop_h));

        // Resize to original image size and threshold
        cv::Mat prob_resized;
        cv::resize(prob_cropped, prob_resized, cv::Size(iImg.cols, iImg.rows), 0, 0, cv::INTER_LINEAR);

        cv::Mat final_mask;
        cv::compare(prob_resized, 0.5f, final_mask, cv::CMP_GT); // CV_8U 0/255

        // Optional cleanup
        int kernel_size = std::max(5, std::min(iImg.cols, iImg.rows) / 100);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
        cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(final_mask, final_mask, cv::MORPH_OPEN, kernel);
        cv::threshold(final_mask, final_mask, 127, 255, cv::THRESH_BINARY);

        // Save mask
        result.masks.push_back(final_mask);
    }
    else
    {
        CONSOLE_BRIDGE_logError("[SAM]: Unexpected mask tensor shape.");
        // Push empty mask to preserve index alignment with boxes (one mask slot per box).
        result.masks.push_back(cv::Mat());
    }
}
